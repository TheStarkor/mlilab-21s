# python3 train.py --model GCN --epochs 5000 --seed 16 --hidden 64 --lr 0.005 --weight_decay 5e-4
# python3 train.py --model GCN --epochs 5000 --seed 72 --hidden 64 --lr 0.005 --weight_decay 5e-4
# python3 train.py --model GCN --epochs 5000 --seed 128 --hidden 64 --lr 0.005 --weight_decay 5e-4
# python3 train.py --model GCN --epochs 5000 --seed 316 --hidden 64 --lr 0.005 --weight_decay 5e-4
# python3 train.py --model GCN --epochs 5000 --seed 512 --hidden 64 --lr 0.005 --weight_decay 5e-4

# python3 train.py --model GAT --epochs 5000 --seed 16 --hidden 8 --n_heads 8 --lr 0.005 --weight_decay 5e-4
# python3 train.py --model GAT --epochs 5000 --seed 72 --hidden 8 --n_heads 8 --lr 0.005 --weight_decay 5e-4
# python3 train.py --model GAT --epochs 5000 --seed 128 --hidden 8 --n_heads 8 --lr 0.005 --weight_decay 5e-4
# python3 train.py --model GAT --epochs 5000 --seed 316 --hidden 8 --n_heads 8 --lr 0.005 --weight_decay 5e-4
# python3 train.py --model GAT --epochs 5000 --seed 512 --hidden 8 --n_heads 8 --lr 0.005 --weight_decay 5e-4

python3 train.py --model GCN --epochs 1000 --seed 128 --hidden 64 --lr 0.005 --weight_decay 5e-4 --dropout 0.5