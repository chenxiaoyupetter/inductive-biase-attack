#! /bin/bash
echo "The GPU device: $1"
#!/bin/bash



wait
python our_attacks.py --attack OurAlgorithm --gpu 0 --batch_size 1 --model_name vit_base_patch16_224 --filename_prefix paper_results --boundary 1
wait
python our_attacks.py --attack OurAlgorithm --gpu 0 --batch_size 1 --model_name pit_b_224 --filename_prefix paper_results --boundary 1
wait
python our_attacks.py --attack OurAlgorithm --gpu 0 --batch_size 1 --model_name visformer_small --filename_prefix paper_results --boundary 1
wait
python our_attacks.py --attack OurAlgorithm --gpu 0 --batch_size 1 --model_name cait_s24_224 --filename_prefix paper_results --boundary 1
wait
python evaluate.py --boundary 1