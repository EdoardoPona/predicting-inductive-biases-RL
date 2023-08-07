

toy_tasks=(toy_1 toy_2 toy_3 toy_5)

for toy in "${toy_tasks[@]}"
do
	python main.py --rate -1 --prop $toy --probe weak --task probing --model toy-transformer 
	python main.py --rate -1 --prop $toy --probe strong --task probing --model toy-transformer 
done

