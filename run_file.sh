#!/bin/bash                                                                                                                        
#SBATCH --job-name=kv_test                                                            
#SBATCH --output=slurm-%j.out                           
#SBATCH --partition=gpu                                                                                                                                                                                                         
#SBATCH --time=01:00:00                   

#SBATCH --gres=gpu:A100:1                
                                                                                                                                                                                                                                                                                      
python Judge/kv_cache.py