#/bin/bash

if [ ! -d results ]; then
	mkdir results
fi

dir_name=results/trial-$(date +%Y-%m-%d:%H:%M:%S)/
mkdir $dir_name

nTests=10

echo "Starting tests"
for i in $(seq 1 32); do
	file=$i"warps.log"
	echo "Testing with $i warps"


	for j in $(seq 0 3); do
		nBible=$((2 ** j))
		inFile=$nBible"bible.txt"

		echo "perf stat -r $nTests $1 $inFile $i 2> $dir_name$nBible"bible_"$file" 

		( perf stat -r $nTests $1 $inFile $i >> /dev/null ) 2> $dir_name$nBible"bible_"$file 
	done
done
