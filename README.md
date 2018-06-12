FB negotiation dataset experiments
==================================

Model training:

    python run_experiment.py --run_dir runs/repro \
                             --learner SupervisedLearner \
                             --train_file data/selection_repro_train.jsons \
                             --validation_file data/selection_repro_val.jsons \
                             --eval_file data/selection_repro_test.jsons \
                             --metrics response_log_likelihood_bits \
                                       response_perplexity \
                                       response_token_perplexity_micro \
                                       selection_log_likelihood_bits \
                                       selection_accuracy \
                                       selection_perplexity

Self-play dialogue simulation:

    python repl.py --run_dir runs/sim_repro \
                   --agent_a FBReproAgent --agent_b FBReproAgent \
                   --load_a runs/repro/model.pkl --load_b runs/repro/model.pkl \
                   --contexts data/selfplay.txt

