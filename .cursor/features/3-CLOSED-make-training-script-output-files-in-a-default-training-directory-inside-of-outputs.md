# Make Training Script Output Files In A Default Training Directory Inside Of Outputs

## Working directory

`~/Desktop/build_llm_karpathy`

## Contents

Right now, output files from training, such as `outputs/build_llm_output_gpt2_training_data_final_merged_50257_2000_test=false_gpt2_gpt2_full_ft_OUTPUT_11242025_215659.txt` Are saved at the root of the outputs directory. I want them by default to instead be saved in a directory inside of outputs called training. So in other words, the training scripts should by default save their output in a file called outputs/training. This would help to separate the different types of output since I already have an output/inference file.

<!-- CLOSED -->
