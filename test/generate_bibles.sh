#/bin/bash

file=$1
cat $file > "1"$file

for i in $(seq 1 4); do
    size=$((2 ** i))

    filename=$size$1
    cat $file > $filename
    cat $file >> $filename

    file=$filename
done;
