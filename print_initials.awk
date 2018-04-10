/[0-9]",/ { lines[NR] = 1 }
/^"/ {
  if (lines[++new_nr] == 1)
    print $0;
}
