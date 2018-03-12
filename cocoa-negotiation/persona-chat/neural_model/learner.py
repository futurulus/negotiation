import numpy as np
import time as tm
import random
import os
import pdb # set_trace()

from cocoa.lib import logstats
from cocoa.lib.bleu import compute_bleu
from neural_model.batcher import DialogueBatcher
from cocoa.pt_model.util import smart_variable, basic_variable
# from cocoa.model.learner import Learner as BaseLearner, add_learner_arguments

from torch.nn import NLLLoss, parameter
from torch.nn.utils import clip_grad_norm
from torch import optim
import torch

def add_learner_arguments(parser):
    parser.add_argument('--optimizer', default='sgd', help='Optimization method')
    parser.add_argument('--sample-targets', action='store_true', help='Sample targets from candidates')
    parser.add_argument('--grad-clip', type=int, default=5, help='Min and max values of gradients')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--min-epochs', type=int, default=10, help='Number of training epochs to run before checking for early stop')
    parser.add_argument('--max-epochs', type=int, default=50, help='Maximum number of training epochs')
    parser.add_argument('--num-per-epoch', type=int, default=None, help='Number of examples per epoch')
    parser.add_argument('--print-every', type=int, default=100, help='Number of examples between printing training loss')
    parser.add_argument('--val-every', type=int, default=140, help='Number of examples between printing validation loss')
    parser.add_argument('--init-from', default="checkpoints/", help='Initial parameters')
    parser.add_argument('--checkpoint', default='pytorch_model', help='Directory to save learned models')
    parser.add_argument('--mappings', default='.', help='Directory to save mappings/vocab')
    parser.add_argument('--gpu', type=int, default=0, help='Use GPU or not')
    parser.add_argument('--summary-dir', default='/tmp', help='Path to summary logs')
    parser.add_argument('--eval-modes', nargs='*', default=('loss',), help='What to evaluate {loss, generation}')

class Learner(object):
    def __init__(self, args, encoder, decoder, vocab):
        self.encoder = encoder
        self.decoder = decoder
        self.enc_optimizer = self.add_optimizers(self.encoder, args)
        self.dec_optimizer = self.add_optimizers(self.decoder, args)

        self.vocab = vocab
        self.create_embeddings()
        self.start_token = vocab.word_to_ind['<s>']
        self.end_token = vocab.word_to_ind['</s>']
        # self.evaluator = evaluator

        self.train_data = DialogueBatcher(vocab, "train")
        self.val_data = DialogueBatcher(vocab, "valid")
        # self.test_data = DialogueBatcher(vocab, "test")

        self.summary_dir = args.summary_dir
        self.verbose = args.verbose
        self.criterion = NLLLoss()
        self.teach_ratio = args.teacher_forcing_ratio
        self.grad_clip = args.grad_clip

        self.train_iterations = self.train_data.num_per_epoch * args.min_epochs
        self.val_iterations = self.train_data.num_per_epoch * args.min_epochs
        self.print_every = args.print_every
        self.val_every = args.val_every

    def _run_batch(self, dialogue_batch, sess, summary_map, test=True):
        raise NotImplementedError

    def test_loss(self, sess, test_data, num_batches):
        '''
        Return the cross-entropy loss.
        '''
        summary_map = {}
        for i in xrange(num_batches):
            dialogue_batch = test_data.next()
            self._run_batch(dialogue_batch, sess, summary_map, test=True)
        return summary_map

    def add_optimizers(self, model, args):
        optimizers = {'adagrad': optim.Adagrad,
                          'sgd': optim.SGD,
                         'adam': optim.Adam}
        optimizer = optimizers[args.optimizer]
        return optimizer(model.parameters(), args.learning_rate)

    def create_embeddings(self):
        # embedding is always tied, can change this to decouple in the future
        vocab_matrix = self.encoder.create_embedding(self.vocab.size)
        self.decoder.create_embedding(vocab_matrix, self.vocab.size)

    def _print_batch(self, batch, preds, loss):
        batcher = self.data.dialogue_batcher
        textint_map = self.data.textint_map
        # Go over each example in the batch
        print '-------------- Batch ----------------'
        for i in xrange(batch['size']):
            success = batcher.print_batch(batch, i, textint_map, preds)
        print 'BATCH LOSS:', loss

    def evaluate(self, batch_val_loss, batch_bleu):
        avg_val_loss = sum(batch_val_loss) * 1.0 / len(batch_val_loss)
        avg_bleu = 100 * float(sum(batch_bleu)) / len(batch_bleu)

        print('Validation Loss: {0:2.4f}, BLEU Score: {1:.2f}'.format(
            avg_val_loss, avg_bleu))
        return avg_val_loss, avg_bleu

    def learn(self, args):
        start = tm.time()
        assert args.min_epochs <= args.max_epochs

        # Gradient
        save_model = False
        train_steps, train_losses = [], []
        val_steps, val_losses = [], []
        bleu_scores, accuracy = [], []

        iters = self.train_iterations
        print_loss_total = 0  # Reset every print_every
        # plot_loss_total = 0  # Reset every plot_every
        # enc_scheduler = StepLR(self.enc_optimizer, step_size=iters/3, gamma=0.2)
        # dec_scheduler = StepLR(self.dec_optimizer, step_size=iters/3, gamma=0.2)

        for iter in range(1, iters + 1):
            # enc_scheduler.step()
            # dec_scheduler.step()
            training_pair = self.train_data.get_batch()
            input_variable = training_pair[0]
            output_variable = training_pair[1]

            loss = self.train(input_variable, output_variable)
            print_loss_total += loss
            # plot_loss_total += loss

            if iter % self.print_every == 0:
                print_loss_avg = print_loss_total / self.print_every
                print_loss_total = 0
                print('{0:3.1f}% complete in {1:.2f} minutes, Train Loss: {2:.4f}'.format(
                    ((iter*100.0) / iters), (tm.time() - start)/60.0, print_loss_avg))
                train_losses.append(print_loss_avg)
                train_steps.append(iter)

            if iter % self.val_every == 0:
                val_steps.append(iter)
                batch_val_loss, batch_bleu = [], []
                for val_iter in range(1, self.val_iterations + 1):
                    val_pair = self.val_data.get_batch()
                    val_input = val_pair[0]
                    val_output = val_pair[1]
                    val_loss, bleu_score = self.validate(val_input, val_output, val_iter)
                    batch_val_loss.append(val_loss)
                    batch_bleu.append(bleu_score)

                avg_val_loss, avg_bleu = self.evaluate(batch_val_loss, batch_bleu)
                val_losses.append(avg_val_loss)
                bleu_scores.append(avg_bleu)

        print('100.0% complete in {0:.2f} minutes, Train Loss: {1:.4f}'.format(
            (tm.time() - start)/60.0, print_loss_avg))

        return train_steps, train_losses, val_steps, val_losses

    def run_inference(self, sources, targets, teach_ratio):
        loss = 0
        encoder_hidden = self.encoder.initHidden()
        encoder_length = sources.size()[0]
        encoder_outputs, encoder_hidden = self.encoder(sources, encoder_hidden)

        decoder_hidden = encoder_hidden
        decoder_length = targets.size()[0]
        decoder_input = basic_variable([self.start_token])
        decoder_context = smart_variable(torch.zeros(1, 1, self.decoder.hidden_size))
        # visual = torch.zeros(encoder_length, decoder_length)
        predictions = []

        for di in range(decoder_length):
            use_teacher_forcing = random.random() < self.teach_ratio
            decoder_output, decoder_context, decoder_hidden, attn_weights = self.decoder(
                decoder_input, decoder_context, decoder_hidden, encoder_outputs)

            # visual[:, di] = attn_weights.squeeze(0).squeeze(0).cpu().data
            loss += self.criterion(decoder_output, targets[di])

            if use_teacher_forcing:
                decoder_input = targets[di]
            else:       # Use the predicted word as the next input
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]
                predictions.append(ni)
                if ni == self.end_token:
                    break
                decoder_input = smart_variable([ni], "list")

        return loss, predictions

    def train(self, input_variable, target_variable):
        self.encoder.train()
        self.decoder.train()
        self.enc_optimizer.zero_grad()
        self.dec_optimizer.zero_grad()

        loss, _ = self.run_inference(input_variable, target_variable, self.teach_ratio)

        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.encoder.parameters(), self.grad_clip)
            clip_grad_norm(self.decoder.parameters(), self.grad_clip)
        self.enc_optimizer.step()
        self.dec_optimizer.step()

        return loss.data[0] / target_variable.size()[0]

    def validate(self, input_variable, target_variable, val_iter):
        self.encoder.eval()  # affects the performance of dropout
        self.decoder.eval()

        loss, predictions = self.run_inference(input_variable, target_variable, teach_ratio=0)
        # queries = input_variable.data.tolist()
        targets = target_variable.data.tolist()
        predicted_tokens = [self.vocab.ind_to_word[x] for x in predictions]
        # query_tokens = [self.vocab.ind_to_word[y] for y in queries]
        target_tokens = [self.vocab.ind_to_word[z] for z in targets]

        if val_iter % 200 == 0:
            print("Target: {}".format(" ".join(target_tokens)))
            print("Predicted: {}".format(" ".join(predicted_tokens)))

        avg_loss = loss.data[0] / target_variable.size()[0]
        bleu_score = compute_bleu(predicted_tokens, target_tokens)
        # turn_success = [pred == tar for pred, tar in zip(predictions, targets)]
        return avg_loss, bleu_score

        # Save model after each epoch
        # print 'Save model checkpoint to', save_path
        # saver.save(sess, save_path, global_step=epoch)

        # Evaluate on dev
        # for split, test_data, num_batches in self.evaluator.dataset():

        #     results = self.eval(sess, split, test_data, num_batches)

        #     # Start to record no improvement epochs
        #     loss = results['loss']
        #     if split == 'dev' and epoch > args.min_epochs:
        #         if loss < best_loss * 0.995:
        #             num_epoch_no_impr = 0
        #         else:
        #             num_epoch_no_impr += 1

        #     if split == 'dev' and loss < best_loss:
        #         print 'New best model'
        #         best_loss = loss
        #         best_saver.save(sess, best_save_path)
        #         self.log_results('best_model', results)
        #         logstats.add('best_model', {'epoch': epoch})

        # # Early stop when no improvement
        # if (epoch > args.min_epochs and num_epoch_no_impr >= 5) or epoch > args.max_epochs:
        #     break
        # epoch += 1

    def save_model(self, args):
        if not os.path.isdir(os.path.dirname(args.init_from)):
            os.makedirs(os.path.dirname(args.init_from))

        enc_path = os.path.join(args.init_from, args.checkpoint+'_encoder.pt')
        torch.save(self.encoder, enc_path)
        print("Saved encoder to {}".format(enc_path))

        dec_path = os.path.join(args.init_from, args.checkpoint+'_decoder.pt')
        torch.save(self.decoder, dec_path)
        print("Saved deccoder to {}".format(dec_path))


    # def log_results(self, name, results):
    #     logstats.add(name, {'loss': results.get('loss', None)})
    #     logstats.add(name, self.evaluator.log_dict(results))