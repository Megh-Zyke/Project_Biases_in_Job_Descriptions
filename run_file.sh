#!/bin/bash                                                                                                                        
#SBATCH --job-name=kv_test                                                            
#SBATCH --output=slurm-%j.out                           
#SBATCH --partition=gpu                                                                                                                                                                                                         
#SBATCH --time=06:00:00                   

#SBATCH --gres=gpu:A100:1                
                                                                                                                                                                                                                                                                                      
python persona_generation/persona_generation.py
