#!/bin/bash

echo "Checking jobs..."
echo

printf "%-12s %-28s %-6s %-25s %s\n" \
    "JOB ID" "JOB NAME" "STATE" "ESTIMATED START" "WAIT TIME"

printf "%-12s %-28s %-6s %-25s %s\n" \
    "------------" "----------------------------" "------" "-------------------------" "----------"

qstat | awk '$1 ~ /^[0-9]/ {print $1}' | while read -r full_jobid
do
    jobid="${full_jobid%%.*}"

    info=$(qstat -f "$jobid" 2>/dev/null)

    jobname=$(echo "$info" \
        | sed -n 's/^[[:space:]]*Job_Name = //p' \
        | head -1)

    state=$(echo "$info" \
        | awk -F' = ' '/^[[:space:]]*job_state =/ {print $2; exit}')

    start_time=$(echo "$info" \
        | sed -n 's/^[[:space:]]*estimated.start_time = //p' \
        | head -1)

    if [ -z "$jobname" ]; then
        jobname="?"
    fi

    if [ -z "$state" ]; then
        state="?"
    fi

    if [ "$state" = "R" ]; then
        start_time="Running now"
        wait_time="0m"

    elif [ -n "$start_time" ]; then
        now_epoch=$(date +%s)
        start_epoch=$(date -d "$start_time" +%s 2>/dev/null)

        if [ -n "$start_epoch" ]; then
            diff=$((start_epoch - now_epoch))

            if [ "$diff" -le 0 ]; then
                wait_time="Due now"
            else
                hours=$((diff / 3600))
                minutes=$(((diff % 3600) / 60))

                if [ "$hours" -gt 0 ]; then
                    wait_time="${hours}h ${minutes}m"
                else
                    wait_time="${minutes}m"
                fi
            fi
        else
            wait_time="?"
        fi

    else
        start_time="Not available"
        wait_time="?"
    fi

    printf "%-12s %-28s %-6s %-25s %s\n" \
        "$jobid" "$jobname" "$state" "$start_time" "$wait_time"
done
