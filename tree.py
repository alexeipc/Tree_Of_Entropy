from multiprocessing import shared_memory
from typing import Any, Optional

import numpy as np
import torch
from vllm import LLM, SamplingParams

from opsd.opsd import OPSD
from tree_reward import TreeRewardManager
from util.debug import debug
from util.jsd_sparse import future_disagreement
from util.reward_func import reward


class Tree:
    """
    Breadth-first tree generation.

    Instead of recursively finishing one branch before moving to another,
    this implementation:

        1. Generates every request in the current layer.
        2. Batches every branching/disagreement probe from that layer.
        3. Pushes all unfinished children into one next-layer queue.
        4. Repeats until the queue is empty.

    This keeps vLLM batches much larger than depth-first recursion.
    """


    def __init__(
        self,
        llm: LLM,
        eos_id: int,
        tree_reward_manager: TreeRewardManager,
        branching_threshold: float = 0.3,
        batch_size: int = 64,
    ):
        self.llm = llm
        self.eos_id = int(eos_id)

        self.n_branch = 2
        self.max_depth = 5
        self.branching_threshold = branching_threshold

        self.manager = tree_reward_manager

        # Maximum number of requests passed to one primary vLLM call.
        # Increase this if your KV cache can hold more requests.
        self.batch_size = batch_size

        # Maximum number of tokens generated after the original input.
        self.max_generated_length = 2048

    # ------------------------------------------------------------------
    # Reward normalization
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_group(groups: dict[Any, list[dict[str, Any]]]) -> None:
        debug("*#" * 80)

        for group in groups.values():
            if not group:
                continue

            rewards = torch.tensor(
                [item["reward"] for item in group],
                dtype=torch.float32,
            )

            advantages = (
                rewards - rewards.mean()
            ) / (
                rewards.std(unbiased=False) + 1e-8
            )

            debug(advantages)

            for item, advantage in zip(group, advantages):
                item["node"].advantage = advantage.item()

        debug("#" * 80)

    # ------------------------------------------------------------------
    # Teacher queue
    # ------------------------------------------------------------------

    @staticmethod
    def _create_teacher_queue() -> dict[str, list[Any]]:
        return {
            "messages": [],
            "reference_answers": [],
            "gts": [],
            "input_ids": [],
        }

    @staticmethod
    def _add_to_teacher_queue(
        teacher_queue: dict[str, list[Any]],
        message: str,
        reference_answer: str,
        gt: str,
        input_ids: torch.Tensor,
    ) -> None:
        teacher_queue["messages"].append(message)
        teacher_queue["reference_answers"].append(reference_answer)
        teacher_queue["gts"].append(gt)
        teacher_queue["input_ids"].append(input_ids)

    def _flush_teacher_queue(
        self,
        teacher_queue: dict[str, list[Any]],
    ) -> None:
        num_requests = len(teacher_queue["messages"])

        if num_requests == 0:
            debug("TEACHER QUEUE IS EMPTY")
            return

        queue_lengths = {
            key: len(values)
            for key, values in teacher_queue.items()
        }

        if len(set(queue_lengths.values())) != 1:
            raise RuntimeError(
                "Teacher queue fields are misaligned: "
                f"{queue_lengths}"
            )

        debug("=" * 80)
        debug(
            f"FLUSHING TEACHER QUEUE: "
            f"{num_requests} REQUESTS"
        )

        leaf_counts = []
        for input_ids in teacher_queue["input_ids"]:
            node = self.manager.add_node(input_ids)
            leaf_counts.append(node.count_branches())

        teacher_n_generations = min(4,max(leaf_counts))
        debug(
            "TEACHER GENERATIONS (MAX STUDENT LEAF COUNT): "
            f"{teacher_n_generations}; LEAF COUNTS: {leaf_counts}"
        )
        debug("=" * 80)

        OPSD.handle_batch(
            vllm=self.llm,
            teacher_messages=teacher_queue["messages"],
            teacher_reference_answers=teacher_queue[
                "reference_answers"
            ],
            teacher_n_generations=teacher_n_generations,
            teacher_gts=teacher_queue["gts"],
            manager=self.manager,
            input_ids=teacher_queue["input_ids"],
            max_tokens=self.max_generated_length,
        )

        for values in teacher_queue.values():
            values.clear()

    # ------------------------------------------------------------------
    # Tensor / completion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_long_cpu_tensor(token_ids) -> torch.Tensor:
        if isinstance(token_ids, torch.Tensor):
            return token_ids.detach().to(
                device="cpu",
                dtype=torch.long,
            )

        return torch.tensor(
            token_ids,
            dtype=torch.long,
            device="cpu",
        )

    def _join_completion(
        self,
        input_ids: torch.Tensor,
        completion,
        remove_last_token: bool,
    ) -> torch.Tensor:
        generated_ids = completion.token_ids

        if remove_last_token:
            generated_ids = generated_ids[:-1]

        generated_ids = self._to_long_cpu_tensor(generated_ids)
        input_ids = self._to_long_cpu_tensor(input_ids)

        return torch.cat(
            [input_ids, generated_ids],
            dim=0,
        )

    def _is_finished(
        self,
        ids: torch.Tensor,
        initial_input_length: int,
    ) -> bool:
        if len(ids) == 0:
            return True

        generated_length = max(
            0,
            len(ids) - initial_input_length,
        )

        return (
            int(ids[-1].item()) == self.eos_id
            or generated_length >= self.max_generated_length
        )

    # ------------------------------------------------------------------
    # Main-layer generation
    # ------------------------------------------------------------------

    def _generate_batch(
        self,
        requests: list[dict[str, Any]],
    ):
        """
        Generate one chunk from the current BFS layer.

        Returns:
            outputs, stop_entropies
        """
        num_requests = len(requests)

        if num_requests == 0:
            return [], []

        entropy_shm = shared_memory.SharedMemory(
            create=True,
            size=num_requests * np.dtype(np.float64).itemsize,
        )

        stop_entropies_shared = np.ndarray(
            (num_requests,),
            dtype=np.float64,
            buffer=entropy_shm.buf,
        )
        stop_entropies_shared[:] = np.nan

        prompts = [request["prompt"] for request in requests]
        sampling_params = []

        for slot, request in enumerate(requests):
            input_ids = request["input_ids"]
            depth = request["depth"]
            threshold = request["threshold"]

            effective_threshold = (
                threshold
                if depth < self.max_depth
                else float("inf")
            )

            initial_input_length = request["initial_input_length"]
            generated_so_far = max(
                0,
                len(input_ids) - initial_input_length,
            )
            remaining_tokens = (
                self.max_generated_length - generated_so_far
            )

            # Requests that have already exhausted the generated-token
            # budget should normally have been saved as leaves before this
            # point. Keep at least one token here only as a defensive guard
            # against invalid SamplingParams.
            max_tokens = max(1, remaining_tokens)

            sampling_params.append(
                SamplingParams(
                    temperature=0.6,
                    top_p=0.95,
                    max_tokens=max_tokens,
                    logprobs=19,
                    extra_args={
                        "entropy_threshold": float(
                            effective_threshold
                        ),
                        "entropy_eos_token_id": self.eos_id,
                        "entropy_shm_name": entropy_shm.name,
                        "entropy_slot": slot,
                        "entropy_num_slots": num_requests,
                    },
                )
            )

        try:
            outputs = self.llm.generate(
                prompts=prompts,
                sampling_params=sampling_params,
            )

            stop_entropies = [
                None if np.isnan(value) else float(value)
                for value in stop_entropies_shared.copy()
            ]
        finally:
            entropy_shm.close()
            entropy_shm.unlink()

        return outputs, stop_entropies

    def _generate_layer(
        self,
        layer_queue: list[dict[str, Any]],
    ):
        """
        Generate the entire current layer.

        The layer may be split into capacity-sized chunks, but no child layer
        starts until every request in this layer has completed.
        """
        all_outputs = []
        all_stop_entropies = []

        for start_idx in range(
            0,
            len(layer_queue),
            self.batch_size,
        ):
            end_idx = min(
                start_idx + self.batch_size,
                len(layer_queue),
            )

            chunk = layer_queue[start_idx:end_idx]

            debug(
                f"GENERATING LAYER CHUNK: "
                f"{start_idx}:{end_idx} / {len(layer_queue)}"
            )

            outputs, stop_entropies = self._generate_batch(chunk)

            all_outputs.extend(outputs)
            all_stop_entropies.extend(stop_entropies)

        if len(all_outputs) != len(layer_queue):
            raise RuntimeError(
                "vLLM output count does not match layer size: "
                f"outputs={len(all_outputs)}, "
                f"layer={len(layer_queue)}"
            )

        return all_outputs, all_stop_entropies

    # ------------------------------------------------------------------
    # Batched branching probes
    # ------------------------------------------------------------------

    def _run_branch_probes(
        self,
        crossing_requests: list[dict[str, Any]],
        force_branch: bool = False,
    ) -> list[list[Any]]:
        """
        Run every future-disagreement probe from the current layer in one
        batched vLLM call.

        Each returned entry has this format:

            [
                current_node_ids,
                (next_prompt, next_ids),
                ...
            ]
        """
        if not crossing_requests:
            return []

        probe_prompts = [
            request["next_prompt"]
            for request in crossing_requests
        ]

        sampling_params = []

        for request in crossing_requests:
            generated_so_far = max(
                0,
                len(request["node_ids"])
                - request["initial_input_length"],
            )
            remaining_tokens = (
                self.max_generated_length - generated_so_far
            )

            sampling_params.append(
                SamplingParams(
                    n=self.n_branch,
                    max_tokens=max(1, min(10, remaining_tokens)),
                    temperature=0.6,
                    top_p=0.95,
                    logprobs=10,
                )
            )

        debug(
            f"RUNNING BATCHED BRANCH PROBES: "
            f"{len(probe_prompts)} REQUESTS"
        )

        probe_outputs = self.llm.generate(
            prompts=probe_prompts,
            sampling_params=sampling_params,
        )

        if len(probe_outputs) != len(crossing_requests):
            raise RuntimeError(
                "Branch-probe output count does not match input count: "
                f"outputs={len(probe_outputs)}, "
                f"requests={len(crossing_requests)}"
            )

        results = []

        for request, output in zip(
            crossing_requests,
            probe_outputs,
        ):
            next_prompt = request["next_prompt"]
            current_node_ids = request["node_ids"]
            branch_completions = output.outputs

            branching_score = future_disagreement(
                branch_completions
            )

            if (
                force_branch
                or branching_score >= self.branching_threshold
            ):
                branch_result = [current_node_ids]

                for branch_completion in branch_completions:
                    branch_ids = self._to_long_cpu_tensor(
                        branch_completion.token_ids
                    )

                    new_ids = torch.cat(
                        [current_node_ids, branch_ids],
                        dim=0,
                    )

                    branch_result.append(
                        (
                            next_prompt + branch_completion.text,
                            new_ids,
                        )
                    )

                results.append(branch_result)

            else:
                # No split, but still advance using one generated probe.
                # Re-queueing current_node_ids unchanged would create an
                # infinite loop because the same prefix can immediately hit
                # the same entropy threshold again.
                if not branch_completions:
                    raise RuntimeError(
                        "Branch probe returned no completions."
                    )

                continuation = branch_completions[0]
                continuation_ids = self._to_long_cpu_tensor(
                    continuation.token_ids
                )

                continued_ids = torch.cat(
                    [current_node_ids, continuation_ids],
                    dim=0,
                )

                if len(continued_ids) <= len(current_node_ids):
                    raise RuntimeError(
                        "Non-branch continuation made no token progress: "
                        f"parent_length={len(current_node_ids)}, "
                        f"child_length={len(continued_ids)}"
                    )

                continued_prompt = (
                    next_prompt + continuation.text
                )

                results.append(
                    [
                        current_node_ids,
                        (continued_prompt, continued_ids),
                    ]
                )

        return results

    # ------------------------------------------------------------------
    # Leaf handling
    # ------------------------------------------------------------------

    def _save_leaf(
        self,
        prompt: str,
        ids: torch.Tensor,
        initial_input_length: int,
        group_id: int,
        parent_ids: Optional[torch.Tensor],
        gts,
        groups: dict[Any, list[dict[str, Any]]],
    ) -> None:
        reward_score = reward(
            prompt,
            gts[group_id],
        )

        node = self.manager.add_node(ids)
        node.correct_answer = reward_score >= 1
        node.wrong_answer = 1 - node.correct_answer
        node.is_leaf = True

        groups.setdefault(group_id, []).append(
            {
                "reward": reward_score,
                "node": node,
                "response_length": max(
                    0,
                    len(ids) - initial_input_length,
                ),
            }
        )

        if parent_ids is not None:
            self.manager.add_child(
                parent_ids=parent_ids,
                child_ids=ids,
            )

    # ------------------------------------------------------------------
    # Breadth-first forward
    # ------------------------------------------------------------------

    def forward(
        self,
        batch_messages,
        depths,
        reference_answers,
        gts,
        groups=None,
        is_init=False,
    ):
        """
        Build the tree breadth-first.

        Every unfinished child produced by the current layer is pushed into
        next_layer_queue. The current layer is fully processed before the
        next layer begins.
        """
        if groups is None:
            groups = {}

        prompts = batch_messages["text"]
        input_ids = batch_messages["input_ids"]
        group_ids = batch_messages["group_ids"]
        thresholds = batch_messages["thresholds"]

        num_roots = len(prompts)

        if not (
            len(input_ids)
            == len(group_ids)
            == len(thresholds)
            == len(depths)
            == num_roots
        ):
            raise ValueError(
                "Initial batch fields must have equal lengths: "
                f"prompts={len(prompts)}, "
                f"input_ids={len(input_ids)}, "
                f"group_ids={len(group_ids)}, "
                f"thresholds={len(thresholds)}, "
                f"depths={len(depths)}"
            )

        teacher_queue = self._create_teacher_queue()

        # Preserve the old behavior: queue teacher generations for all
        # original prompts when this is the initialization call.
        if is_init:
            for i, prompt in enumerate(prompts):
                group_id = group_ids[i]

                self._add_to_teacher_queue(
                    teacher_queue=teacher_queue,
                    message=prompt,
                    reference_answer=reference_answers[group_id],
                    gt=gts[group_id],
                    input_ids=self._to_long_cpu_tensor(
                        input_ids[i]
                    ),
                )

        # return_node_ids[i] is the first generated tree node corresponding
        # to original root request i.
        return_node_ids: list[Optional[torch.Tensor]] = [
            None
        ] * num_roots

        current_layer_queue = []

        if is_init:
            # Force a split directly at each original prompt. This root
            # split is a real tree level, so its children start at depth + 1.
            root_requests = []
            for i in range(num_roots):
                root_ids = self._to_long_cpu_tensor(input_ids[i])
                root_requests.append(
                    {
                        "node_ids": root_ids,
                        "next_prompt": prompts[i],
                        "group_id": group_ids[i],
                        "threshold": thresholds[i],
                        "depth": depths[i],
                        "initial_input_length": len(input_ids[i]),
                    }
                )

            root_branch_results = self._run_branch_probes(
                root_requests,
                force_branch=True,
            )

            for root_index, (root, branch_result) in enumerate(
                zip(root_requests, root_branch_results)
            ):
                parent_node_ids = branch_result[0]
                next_depth = root["depth"] + 1

                for branch_index, (next_prompt, child_ids) in enumerate(
                    branch_result[1:]
                ):
                    if self._is_finished(
                        child_ids,
                        root["initial_input_length"],
                    ):
                        self._save_leaf(
                            prompt=next_prompt,
                            ids=child_ids,
                            initial_input_length=root[
                                "initial_input_length"
                            ],
                            group_id=root["group_id"],
                            parent_ids=parent_node_ids,
                            gts=gts,
                            groups=groups,
                        )
                        if branch_index == 0:
                            return_node_ids[root_index] = child_ids
                        continue

                    current_layer_queue.append(
                        {
                            "prompt": next_prompt,
                            "input_ids": child_ids,
                            "group_id": root["group_id"],
                            "threshold": root["threshold"],
                            "depth": next_depth,
                            "parent_ids": parent_node_ids,
                            "return_slot": (
                                root_index if branch_index == 0 else None
                            ),
                            "initial_input_length": root[
                                "initial_input_length"
                            ],
                        }
                    )
        else:
            for i in range(num_roots):
                current_layer_queue.append(
                    {
                        "prompt": prompts[i],
                        "input_ids": self._to_long_cpu_tensor(
                            input_ids[i]
                        ),
                        "group_id": group_ids[i],
                        "threshold": thresholds[i],
                        "depth": depths[i],
                        "parent_ids": None,
                        "return_slot": i,
                        "initial_input_length": len(input_ids[i]),
                    }
                )

        layer_number = 0

        while current_layer_queue:
            debug("=" * 80)
            debug(
                f"BFS LAYER {layer_number}: "
                f"{len(current_layer_queue)} REQUESTS"
            )
            debug("=" * 80)

            outputs, stop_entropies = self._generate_layer(
                current_layer_queue
            )

            next_layer_queue = []

            # Requests that crossed the entropy threshold are collected
            # first, then all disagreement probes are generated together.
            crossing_requests = []

            for request, output, stop_entropy in zip(
                current_layer_queue,
                outputs,
                stop_entropies,
            ):
                completion = output.outputs[0]

                prompt = request["prompt"]
                old_input_ids = request["input_ids"]
                group_id = request["group_id"]
                parent_ids = request["parent_ids"]
                return_slot = request["return_slot"]
                initial_input_length = request["initial_input_length"]

                if stop_entropy is None:
                    # EOS or max token limit: save a completed leaf.
                    node_ids = self._join_completion(
                        input_ids=old_input_ids,
                        completion=completion,
                        remove_last_token=False,
                    )

                    full_prompt = prompt + completion.text

                    self._save_leaf(
                        prompt=full_prompt,
                        ids=node_ids,
                        initial_input_length=initial_input_length,
                        group_id=group_id,
                        parent_ids=parent_ids,
                        gts=gts,
                        groups=groups,
                    )

                    if return_slot is not None:
                        return_node_ids[return_slot] = node_ids

                    continue

                # Threshold crossing: remove the custom stopping token from
                # the stored node IDs, matching the original implementation.
                node_ids = self._join_completion(
                    input_ids=old_input_ids,
                    completion=completion,
                    remove_last_token=True,
                )

                if parent_ids is not None:
                    self.manager.add_child(
                        parent_ids=parent_ids,
                        child_ids=node_ids,
                    )

                self.manager.add_node(node_ids)

                if return_slot is not None:
                    return_node_ids[return_slot] = node_ids

                next_prompt = prompt + completion.text

                self._add_to_teacher_queue(
                    teacher_queue=teacher_queue,
                    message=next_prompt,
                    reference_answer=reference_answers[group_id],
                    gt=gts[group_id],
                    input_ids=node_ids,
                )

                crossing_requests.append(
                    {
                        "node_ids": node_ids,
                        "next_prompt": next_prompt,
                        "group_id": group_id,
                        "threshold": stop_entropy,
                        "depth": request["depth"],
                        "initial_input_length": initial_input_length,
                    }
                )

            # One vLLM call for all branching probes in this BFS layer.
            branch_results = self._run_branch_probes(
                crossing_requests
            )

            for crossing, branch_result in zip(
                crossing_requests,
                branch_results,
            ):
                parent_node_ids = branch_result[0]
                group_id = crossing["group_id"]

                actual_branch = len(branch_result) > 2
                next_depth = crossing["depth"] + int(
                    actual_branch
                )

                for next_prompt, child_ids in branch_result[1:]:
                    if len(child_ids) <= len(parent_node_ids):
                        raise RuntimeError(
                            "Queued child made no token progress: "
                            f"parent_length={len(parent_node_ids)}, "
                            f"child_length={len(child_ids)}"
                        )

                    if self._is_finished(
                        child_ids,
                        crossing["initial_input_length"],
                    ):
                        self._save_leaf(
                            prompt=next_prompt,
                            ids=child_ids,
                            initial_input_length=crossing[
                                "initial_input_length"
                            ],
                            group_id=group_id,
                            parent_ids=parent_node_ids,
                            gts=gts,
                            groups=groups,
                        )

                        debug("NODE ADDED")
                        continue

                    # Push every unfinished child into the same next-layer
                    # queue. No child is generated until the full current
                    # layer has been processed.
                    next_layer_queue.append(
                        {
                            "prompt": next_prompt,
                            "input_ids": child_ids,
                            "group_id": group_id,
                            "threshold": crossing["threshold"],
                            "depth": next_depth,
                            "parent_ids": parent_node_ids,
                            "return_slot": None,
                            "initial_input_length": crossing[
                                "initial_input_length"
                            ],
                        }
                    )

            debug(
                f"BFS LAYER {layer_number} COMPLETE; "
                f"QUEUED {len(next_layer_queue)} REQUESTS "
                f"FOR LAYER {layer_number + 1}"
            )

            current_layer_queue = next_layer_queue
            layer_number += 1

        missing_slots = [
            i
            for i, node_ids in enumerate(return_node_ids)
            if node_ids is None
        ]

        if missing_slots:
            raise RuntimeError(
                "Some initial requests did not produce a root node: "
                f"{missing_slots}"
            )

        debug(
            "TOTAL QUEUED TEACHER REQUESTS: "
            f"{len(teacher_queue['messages'])}"
        )

        # Generate every teacher request only after the whole student tree
        # has been constructed.
        self._flush_teacher_queue(teacher_queue)

        if is_init:
            Tree.normalize_group(groups)

        return return_node_ids
