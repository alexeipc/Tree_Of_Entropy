# ============================================================
# GPU cluster overview for Slurm partitions: aisc and aiscii
#
# Shows:
#   - partition config
#   - node-level GPU/CPU/memory usage
#   - running jobs
#   - pending jobs
#   - Slurm ACCOUNT used by each job, e.g. --account=aiscii-conf
#   - PRIORITY from scontrol, e.g. Priority=4462
#   - SCHED_NODE only for PENDING jobs, e.g. SchedNodeList=aisciit01
#   - per-user/account totals
#
# Usage:
#   show_gpu_clusters
#   show_gpu_clusters aisc
#   show_gpu_clusters aiscii
#   show_gpu_clusters aisc aiscii
#
# Optional:
#   job_detail <JOBID>
#   show_my_slurm_accounts
# ============================================================

show_gpu_clusters() {
    local parts=("$@")
    if [ ${#parts[@]} -eq 0 ]; then
        parts=(aisc aiscii)
    fi

    if ! command -v squeue >/dev/null 2>&1; then
        echo "ERROR: squeue not found. Are you on a Slurm login node?"
        return 1
    fi

    if ! command -v sinfo >/dev/null 2>&1; then
        echo "ERROR: sinfo not found. Are you on a Slurm login node?"
        return 1
    fi

    echo
    echo "============================================================"
    echo "GPU CLUSTER OVERVIEW"
    echo "Generated at: $(date)"
    echo "Partitions: ${parts[*]}"
    echo "============================================================"

    show_partition_config() {
        local part="$1"

        echo
        echo "PARTITION CONFIG SUMMARY [$part]"

        sinfo -h -p "$part" -o "%P|%a|%l|%D|%t|%G|%f" | \
        awk -F'|' '
        BEGIN {
            count = 0
            printf "%-12s %-8s %-12s %-8s %-12s %-40s %-25s\n", \
                   "PARTITION", "AVAIL", "MAX_TIME", "NODES", "STATE", "GRES", "FEATURES"
        }
        {
            count++
            printf "%-12s %-8s %-12s %-8s %-12s %-40s %-25s\n", \
                   $1, $2, $3, $4, $5, $6, $7
        }
        END {
            if (count == 0) print "(none)"
        }'
    }

    show_node_status() {
        local part="$1"

        echo
        echo "NODE RESOURCE STATUS [$part]"

        local nodes
        nodes=$(sinfo -h -N -p "$part" -o "%N" | sort -u)

        if [ -z "$nodes" ]; then
            echo "(no nodes found from sinfo)"
            return
        fi

        {
            while read -r node; do
                [ -n "$node" ] && scontrol show node -o "$node" 2>/dev/null
            done <<< "$nodes"
        } | \
        awk '
        function getval(k,    r, v) {
            r = k "=[^ ]+"
            if (match($0, r)) {
                v = substr($0, RSTART + length(k) + 1, RLENGTH - length(k) - 1)
                return v
            }
            return "-"
        }

        function gpu_from_gres(gres,    a, i, tok, tmp, total, n) {
            total = 0
            if (gres == "" || gres == "-" || gres == "(null)" || gres == "N/A") return 0

            n = split(gres, a, ",")
            for (i = 1; i <= n; i++) {
                tok = a[i]
                if (tok ~ /gpu/) {
                    sub(/\(.*/, "", tok)
                    if (match(tok, /gpu(:[^:(),]+)*:[0-9]+/)) {
                        tmp = substr(tok, RSTART, RLENGTH)
                        sub(/.*:/, "", tmp)
                        total += tmp + 0
                    }
                }
            }
            return total
        }

        function gpu_from_tres(tres,    a, i, tok, tmp, total, n) {
            total = 0
            if (tres == "" || tres == "-" || tres == "(null)" || tres == "N/A") return 0

            n = split(tres, a, ",")
            for (i = 1; i <= n; i++) {
                tok = a[i]
                if (tok ~ /^gres\/gpu/) {
                    tmp = tok
                    sub(/.*=/, "", tmp)
                    total += tmp + 0
                }
            }
            return total
        }

        function mem_gb(x) {
            if (x == "-" || x == "" || x == "(null)") return "-"
            return sprintf("%.0fG", x / 1024)
        }

        BEGIN {
            count = 0
            total_gpu = 0
            alloc_gpu = 0
            total_cpu = 0
            alloc_cpu = 0
            total_mem = 0
            alloc_mem = 0

            printf "%-15s %-18s %-11s %-11s %-13s %-17s %-40s\n", \
                   "NODE", "STATE", "GPU_USED", "GPU_FREE", "CPU_USED", "MEM_USED", "GRES"
        }

        NF > 0 {
            count++

            node      = getval("NodeName")
            state     = getval("State")
            gres      = getval("Gres")
            gresused  = getval("GresUsed")
            cfgtres   = getval("CfgTRES")
            alloctres = getval("AllocTRES")
            cpualloc  = getval("CPUAlloc")
            cputot    = getval("CPUTot")
            memreal   = getval("RealMemory")
            memalloc  = getval("AllocMem")

            gt = gpu_from_gres(gres)
            if (gt == 0) gt = gpu_from_tres(cfgtres)

            ga = gpu_from_tres(alloctres)
            if (ga == 0) ga = gpu_from_gres(gresused)

            gf = gt - ga
            if (gf < 0) gf = 0

            printf "%-15s %-18s %4d/%-6d %4d/%-6d %5s/%-7s %7s/%-9s %-40s\n", \
                   node, state, ga, gt, gf, gt, cpualloc, cputot, mem_gb(memalloc), mem_gb(memreal), gres

            total_gpu += gt
            alloc_gpu += ga
            total_cpu += cputot + 0
            alloc_cpu += cpualloc + 0
            total_mem += memreal + 0
            alloc_mem += memalloc + 0
        }

        END {
            if (count == 0) {
                print "(none)"
                print "Note: sinfo found nodes, but scontrol returned no node details."
            } else {
                print "--------------------------------------------------------------------------"
                printf "TOTAL: nodes=%d | GPUs=%d/%d used, %d free | CPUs=%d/%d used | MEM=%s/%s used\n", \
                       count, alloc_gpu, total_gpu, total_gpu - alloc_gpu, \
                       alloc_cpu, total_cpu, mem_gb(alloc_mem), mem_gb(total_mem)
            }
        }'
    }

    show_jobs_by_state() {
        local part="$1"
        local state="$2"
        local title="$3"

        echo
        echo "$title [$part]"

        {
            squeue -h -t "$state" -p "$part" \
                -o "%i|%u|%a|%P|%j|%T|%V|%S|%N|%D|%b|%C|%m|%l|%M|%L|%R" | \
            while IFS='|' read -r jid user account partition jobname jobstate submit start nodelist nnodes gres cpus mem tlimit used left reason; do
                [ -z "$jid" ] && continue

                jobinfo=$(scontrol show job -o "$jid" 2>/dev/null || true)

                priority=$(printf "%s\n" "$jobinfo" | sed -n 's/.* Priority=\([^ ]*\).*/\1/p')
                schednode=$(printf "%s\n" "$jobinfo" | sed -n 's/.* SchedNodeList=\([^ ]*\).*/\1/p')

                [ -z "$priority" ] && priority="-"
                [ -z "$schednode" ] && schednode="-"
                [ "$schednode" = "(null)" ] && schednode="-"

                printf "%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n" \
                    "$jid" "$user" "$account" "$partition" "$jobname" "$jobstate" \
                    "$submit" "$start" "$nodelist" "$nnodes" "$gres" "$cpus" "$mem" \
                    "$tlimit" "$used" "$left" "$reason" "$priority" "$schednode"
            done
        } | \
        awk -F'|' -v want_state="$state" '
        function to_epoch(s,    t) {
            if (s == "" || s == "N/A" || s == "Unknown" || s == "(null)") return 0
            t = s
            sub(/\..*/, "", t)
            gsub(/T/, " ", t)
            gsub(/[-:]/, " ", t)
            return mktime(t)
        }

        function fmt_dur(sec,    d, h, m, s) {
            if (sec < 0) sec = 0
            d = int(sec / 86400)
            h = int((sec % 86400) / 3600)
            m = int((sec % 3600) / 60)
            s = sec % 60

            if (d > 0) return sprintf("%dd %02d:%02d:%02d", d, h, m, s)
            return sprintf("%02d:%02d:%02d", h, m, s)
        }

        function clean_time(s) {
            if (s == "" || s == "(null)") return "-"
            gsub(/T/, " ", s)
            sub(/\..*/, "", s)
            return s
        }

        function short(s, n) {
            if (s == "" || s == "(null)") return "-"
            if (length(s) > n) return substr(s, 1, n - 1) "+"
            return s
        }

        function gpu_count(gres,    a, i, tok, tmp, total, n) {
            total = 0
            if (gres == "" || gres == "N/A" || gres == "(null)") return "-"

            n = split(gres, a, ",")
            for (i = 1; i <= n; i++) {
                tok = a[i]
                if (tok ~ /gpu/) {
                    sub(/\(.*/, "", tok)
                    if (match(tok, /gpu(:[^:=,]+)*[:=][0-9]+/)) {
                        tmp = substr(tok, RSTART, RLENGTH)
                        sub(/.*[:=]/, "", tmp)
                        total += tmp + 0
                    }
                }
            }

            return total > 0 ? total : "-"
        }

        BEGIN {
            count = 0
            now = systime()

            if (want_state == "PD") {
                printf "%-9s %-9s %-24s %-10s %-20s %-9s %-8s %-14s %4s %5s %7s %3s %-19s %-19s %-10s %-10s %-10s %-12s %-30s\n", \
                       "JOBID", "USER", "ACCOUNT", "PARTITION", "JOBNAME", "STATE", \
                       "PRIOR", "SCHED_NODE", "GPU", "CPU", "MEM", "N", \
                       "SUBMITTED", "START/EST_START", "TLIMIT", "USED", "LEFT", \
                       "Q_WAIT", "REASON"
            } else {
                printf "%-9s %-9s %-24s %-10s %-20s %-9s %-8s %4s %5s %7s %3s %-19s %-19s %-10s %-10s %-10s %-12s %-30s\n", \
                       "JOBID", "USER", "ACCOUNT", "PARTITION", "JOBNAME", "STATE", \
                       "PRIOR", "GPU", "CPU", "MEM", "N", \
                       "SUBMITTED", "START_TIME", "TLIMIT", "USED", "LEFT", \
                       "Q_WAIT", "NODELIST"
            }
        }

        {
            count++

            submit_epoch = to_epoch($7)
            start_epoch  = to_epoch($8)

            if (want_state == "R") {
                if (submit_epoch > 0 && start_epoch > 0) qwait = fmt_dur(start_epoch - submit_epoch)
                else qwait = "-"
                node_or_reason = $9
            } else {
                if (submit_epoch > 0) qwait = fmt_dur(now - submit_epoch)
                else qwait = "-"
                node_or_reason = $17
            }

            if (want_state == "PD") {
                printf "%-9s %-9s %-24s %-10s %-20s %-9s %-8s %-14s %4s %5s %7s %3s %-19s %-19s %-10s %-10s %-10s %-12s %-30s\n", \
                       $1, short($2, 9), short($3, 24), short($4, 10), \
                       short($5, 20), short($6, 9), short($18, 8), short($19, 14), \
                       gpu_count($11), $12, $13, $10, \
                       short(clean_time($7), 19), short(clean_time($8), 19), \
                       $14, $15, $16, qwait, short(node_or_reason, 30)
            } else {
                printf "%-9s %-9s %-24s %-10s %-20s %-9s %-8s %4s %5s %7s %3s %-19s %-19s %-10s %-10s %-10s %-12s %-30s\n", \
                       $1, short($2, 9), short($3, 24), short($4, 10), \
                       short($5, 20), short($6, 9), short($18, 8), \
                       gpu_count($11), $12, $13, $10, \
                       short(clean_time($7), 19), short(clean_time($8), 19), \
                       $14, $15, $16, qwait, short(node_or_reason, 30)
            }
        }

        END {
            if (count == 0) print "(none)"
        }'
    }

    show_user_totals() {
        local part="$1"

        echo
        echo "PER-USER / ACCOUNT JOB TOTALS [$part]"

        squeue -h -p "$part" -t R,PD -o "%u|%a|%T|%b|%C" | \
        awk -F'|' '
        function gpu_count(gres,    a, i, tok, tmp, total, n) {
            total = 0
            if (gres == "" || gres == "N/A" || gres == "(null)") return 0

            n = split(gres, a, ",")
            for (i = 1; i <= n; i++) {
                tok = a[i]
                if (tok ~ /gpu/) {
                    sub(/\(.*/, "", tok)
                    if (match(tok, /gpu(:[^:=,]+)*[:=][0-9]+/)) {
                        tmp = substr(tok, RSTART, RLENGTH)
                        sub(/.*[:=]/, "", tmp)
                        total += tmp + 0
                    }
                }
            }
            return total
        }

        BEGIN {
            count = 0
            printf "%-10s %-24s %-10s %-8s %-8s %-8s\n", \
                   "USER", "ACCOUNT", "STATE", "JOBS", "GPUS", "CPUS"
        }

        {
            count++
            key = $1 SUBSEP $2 SUBSEP $3
            jobs[key]++
            gpus[key] += gpu_count($4)
            cpus[key] += $5 + 0
        }

        END {
            if (count == 0) {
                print "(none)"
            } else {
                for (key in jobs) {
                    split(key, a, SUBSEP)
                    printf "%-10s %-24s %-10s %-8d %-8d %-8d\n", \
                           a[1], a[2], a[3], jobs[key], gpus[key], cpus[key]
                }
            }
        }'
    }

    for part in "${parts[@]}"; do
        echo
        echo "============================================================"
        echo "PARTITION: $part"
        echo "============================================================"

        show_partition_config "$part"
        show_node_status "$part"
        show_jobs_by_state "$part" "R"  "RUNNING JOBS"
        show_jobs_by_state "$part" "PD" "PENDING JOBS"
        show_user_totals "$part"

        echo
        echo "------------------------------------------------------------"
    done
}

job_detail() {
    if [ -z "$1" ]; then
        echo "Usage: job_detail <JOBID>"
        return 1
    fi

    scontrol show job -dd "$1" | tr ' ' '\n' | grep -E \
'^(JobId|JobName|UserId|GroupId|Account|Partition|JobState|Reason|Priority|Nice|SubmitTime|EligibleTime|StartTime|EndTime|TimeLimit|RunTime|TimeMin|NumNodes|NumCPUs|NumTasks|CPUs/Task|ReqTRES|AllocTRES|TRES|TresPerNode|TresPerTask|MinMemoryNode|MinMemoryCPU|ReqNodeList|ExcNodeList|NodeList|SchedNodeList|BatchHost|SubmitHost|WorkDir|Command|StdOut|StdErr)='
}

show_my_slurm_accounts() {
    echo
    echo "MY SLURM ASSOCIATIONS / ACCOUNTS"

    if command -v sacctmgr >/dev/null 2>&1; then
        sacctmgr -n -P show assoc user="$USER" format=User,Account,Partition,Share,QOS 2>/dev/null | \
        awk -F'|' '
        BEGIN {
            count = 0
            printf "%-12s %-24s %-16s %-8s %-20s\n", "USER", "ACCOUNT", "PARTITION", "SHARE", "QOS"
        }
        NF > 0 {
            count++
            printf "%-12s %-24s %-16s %-8s %-20s\n", $1, $2, $3, $4, $5
        }
        END {
            if (count == 0) print "(none or permission denied)"
        }'
    else
        echo "sacctmgr not available."
    fi
}

# Run immediately for both GPU partitions
show_gpu_clusters aisc aiscii