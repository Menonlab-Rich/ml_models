squeue -t PD -o "%.18i %.9P %.8j %.8u %.2t %.10M %.10L %.6D %R %.12p" --sort=-p | awk -v user="u0977428" '
BEGIN { line=0; all_lines=0; }
/user/ { if (line == 0) line = NR; all_lines++; }
END { if (line == 0) print "No jobs found for user: " user; else print line "/" all_lines; }
'