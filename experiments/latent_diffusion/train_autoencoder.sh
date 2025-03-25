# train autoencoder

# bigger encoding size
python train_autoencoder.py --train_batch_size 4096 --num_epochs 500 --learning_rate 5e-4  --encoding_size 100 --encoding_base 2

# bigger encoding size
# python train_autoencoder.py --train_batch_size 4096 --num_epochs 200 --learning_rate 5e-4  --encoding_size 50 --encoding_base 4

# smaller encoding size
# python train_autoencoder.py --train_batch_size 4096 --num_epochs 400 --learning_rate 5e-4  --encoding_size 5 --encoding_base 4
