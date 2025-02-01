accelerate launch --config_file config.yaml train_graph_llm.py --num_epochs 15 --dataset bgm

accelerate launch --config_file config.yaml train_graph_llm.py --num_epochs 20 --dataset mts

accelerate launch --config_file config.yaml train_graph_llm.py --num_epochs 15 --dataset sc

accelerate launch --config_file config.yaml train_graph_llm.py --num_epochs 20 --dataset sp
