# train conditional autoencoder

# smaller encoding size
python train_conditional_autoencoder.py --conditions 'mean_iota' 'aspect_ratio' 'nfp' 'helicity' --train_batch_size 4096 --num_epochs 200 --learning_rate 5e-4  --encoding_size 5 --encoding_base 4 --cond_emb_size 64
