import __init__

import torch
import time
import numpy as np
import matplotlib.pyplot as plt


def gen_some_text(model, tokenizer, device, max_len_context,
                  text_prompt='The dog ran across the',
                  tokens_to_gen=10,
                  vis=False,
                  decode_style='greedy',
                  decode_seed=0,
                  decode_beta=1.0,
                  decode_sample_topp_threshold=0.70):
    """
    dummy_token: if text_prompt is < max_len_context tokens, need to add dummy tokens and mask them
    1) tokenize the text prompt. TODO [jeremy]: I don't think this is true anymore.

    If vis: plot distribution over vocab for next word, given the past max_len_context
    """
    np.random.seed(decode_seed)

    total_text_string = text_prompt  # this will be extended by tokens_to_gen

    def tokenize_some_text(text='The dog ran across the'):
        """
        Uses tokenizer trained in Dataloader to decode text.
        """
        tokenized_text_ints = torch.tensor(tokenizer(text)['input_ids'][:-1], dtype=torch.long)
        return tokenized_text_ints

    def process_prompt(dummy_token=0):
        # Two cases:
        # - if less than max_len_context (context length), need to add dummy tokens
        # - if longer than max_len_context (context length), truncate to max_len_context
        tokenized_text = tokenize_some_text(text=text_prompt)
        nn = tokenized_text.shape[0]

        if nn > max_len_context:  # take last max_len_context elements
            input_slice = tokenized_text[nn - max_len_context:]
            src_mask = model.generate_square_subsequent_mask(max_len_context).to(device)
        else:
            input_slice = tokenized_text[0:nn]
            src_mask = model.generate_square_subsequent_mask(nn).to(device)

        src = torch.zeros((min(nn, max_len_context), 1), dtype=torch.long)
        src[0:nn, 0] = input_slice
        src.to(device)

        return src, src_mask

    def decode(model_out, style='greedy'):
        """
        Sampling notes:
        - working in log space to improve numeric stability
        - use "Gumbel-max trick"
          - https://stats.stackexchange.com/questions/64081/how-do-i-sample-from-a-discrete-categorical-distribution-in-log-space
          - https://en.wikipedia.org/wiki/Categorical_distribution#Sampling_via_the_Gumbel_distribution
        TODO - verify that this trick works the same as the classic way, for stability we need to compare with high noise (low beta) -- how to make the random samples identical?
        """
        assert style in ['greedy', 'sample_full', 'sample_topp']
        if style == 'greedy':
            guessed_int = torch.argmax(model_out[nn - 1, 0])  # care batch dimension, nn vs max_len_context
        elif style == 'sample_full':
            """
            next_word_weights = out[nn-1, 0].detach().numpy()
            next_word_weights_exp = np.exp(decode_beta * next_word_weights)
            print('exp check')
            print(next_word_weights_exp)
            next_word_probs = next_word_weights_exp / np.sum(next_word_weights_exp)
            print("LIMITS::::::::::::::::", np.min(next_word_probs), np.max(next_word_probs))
            print("LIMITS::::::::::::::::", np.min(next_word_weights_exp), np.max(next_word_weights_exp))
            print("LIMITS::::::::::::::::", np.sum(next_word_weights_exp))
            unirand = np.random.rand()
            distribution_cumsum = np.cumsum(next_word_probs)
            guessed_int = np.searchsorted(distribution_cumsum, unirand)
            print('BBBBBBBBBBBBBBBBBBBBBBBBBBB', type(unirand), unirand)
            print(distribution_cumsum)
            print(guessed_int)
            print(distribution_cumsum[guessed_int-1], distribution_cumsum[guessed_int], distribution_cumsum[guessed_int+1])
            print()"""

            # numerically stable approach: gumbel max-trick sampling
            next_word_weights = out[nn - 1, 0].detach().numpy()
            ncategories = next_word_weights.shape[0]
            next_word_weights_scaled = decode_beta * next_word_weights
            # print("ncategories", ncategories)
            uvec = np.random.rand(ncategories)
            gvec = -np.log(-np.log(uvec))
            guessed_int = np.argmax(gvec + next_word_weights_scaled)
        else:
            assert style == 'sample_topp'
            # TODO implement gumbel max trick here too
            next_word_weights = out[nn - 1, 0].detach().numpy()
            next_word_weights_exp = np.exp(decode_beta * next_word_weights)
            next_word_probs = next_word_weights_exp / np.sum(next_word_weights_exp)
            # 1) identify top p words such their cumulative probability passes threshold
            distribution_sorted_indices = np.argsort(next_word_probs)[::-1]
            next_word_probs_descsort = next_word_probs[distribution_sorted_indices]
            next_word_probs_descsort_cumsum = np.cumsum(next_word_probs_descsort)
            threshold_index = np.searchsorted(next_word_probs_descsort_cumsum,
                                              decode_sample_topp_threshold)
            # 2) sample from these top p words
            topp_indices = distribution_sorted_indices[:threshold_index + 1]
            # print(distribution_sorted_indices)
            # print(next_word_probs_descsort_cumsum)
            # print("topp_indices", len(topp_indices), topp_indices)
            topp_probs = next_word_probs_descsort[:threshold_index + 1]
            topp_reweighted_probs = topp_probs / np.sum(topp_probs)
            topp_reweighted_cumsum = np.cumsum(topp_reweighted_probs)
            unirand = np.random.rand()
            topp_choice = np.searchsorted(topp_reweighted_cumsum, unirand)
            guessed_int = topp_indices[topp_choice]
        # print(model_out[nn - 1, 0])
        return guessed_int

    # 1) tokenize the text prompt and prepare associated src_mask for model.forward()
    src, src_mask = process_prompt(tokenizer.get_vocab()["<pad>"])  # src should be in form ntokens x nbatches
    nn = src_mask.shape[0]
    src.reshape((nn, 1))

    # 2)
    model.eval()
    for idx in range(tokens_to_gen):
        running_context_string = ' '.join([tokenizer.decode(src[k]) for k in range(src.shape[0])])
        # print(running_context_string)
        # TESTING DIFFERENT src_mask (all zero)
        use_diff_mask = False
        if use_diff_mask:
            print('Warning: testing maskless generation')
            # A:
            # src_mask = torch.zeros(nn, nn)
            # B:
            src_mask = torch.zeros(nn, nn) + float('-inf')
            src_mask[:, 0] = 0
            src_mask[-1, :] = 0
            # C:
            # src_mask = torch.rand(nn, nn)
            # print(src_mask)

        # TODO : add src_key_padding_mask to the forward call
        out = model.forward(src, src_mask)
        # print(out.shape)
        if vis:
            next_word_weights = out[nn - 1, 0].detach().numpy()
            next_word_weights_exp = np.exp(next_word_weights)
            next_word_probs = next_word_weights_exp / np.sum(next_word_weights_exp)

            plt.plot(next_word_weights)
            plt.title('next_word_weights: iteration %d' % idx)
            plt.xlabel('vocab index')
            plt.ylabel('weight')
            # plt.xlim((-0.5,100.5))
            plt.show()

            plt.plot(next_word_probs)
            plt.title('next_word_probs: iteration %d' % idx)
            plt.xlabel('vocab index')
            plt.ylabel('weight')
            # plt.xlim((-0.5,100.5))
            plt.show()

            kk = 20
            top_word_indices = np.argsort(next_word_probs)[::-1]
            top_word_probs = next_word_probs[top_word_indices]
            x = list(range(kk))
            plt.title('next_word_probs: iteration %d' % idx)
            plt.suptitle(running_context_string + ' ???', fontsize=8, wrap=True)
            plt.xticks(x, [tokenizer.decode(k) for k in top_word_indices[0:kk]], rotation=60)
            plt.plot(x, top_word_probs[0:kk])
            plt.ylabel('probability')
            plt.show()

        next_guess_int = decode(out, style=decode_style)

        next_guess_string = tokenizer.decode(int(next_guess_int))

        # update total_text_string by adding best guess
        total_text_string += ' %s' % next_guess_string

        # print(total_text_string)

        # update src and src mask for next pass of model.forward()
        if nn < max_len_context:
            # extend and shift the running window of input data
            src_updated = torch.zeros((nn + 1, 1), dtype=torch.long)  # extend src to nn
            src_updated[0:nn, 0] = src[:, 0]
            src_updated[nn, 0] = next_guess_int
            src = src_updated.to(device)
            # increment mask dimension
            src_mask = model.generate_square_subsequent_mask(nn + 1).to(device)
            nn += 1
        else:
            src_orig = src.clone()
            src[0:max_len_context - 1, 0] = src_orig[1:, 0]
            src[-1, 0] = next_guess_int

    return total_text_string


def gen_some_text_wrapper(generator_model, tokenizer, device, text_prompt,
                          decode_style,
                          decode_seed=0,
                          decode_beta=1.0):
    generated_text = gen_some_text(generator_model, tokenizer, device,
                                   text_prompt=text_prompt,
                                   tokens_to_gen=25,
                                   decode_style=decode_style,
                                   decode_seed=decode_seed,
                                   decode_beta=decode_beta,
                                   vis=False)
    return generated_text


def decode_during_training(generator_model, tokenizer, device, epoch,
                           nongreedy_style,
                           text_prompt='The dog ran',
                           decode_seeds=[0, 1, 2],
                           decode_betas=[1.0, 1.0, 1.0]):
    assert len(decode_seeds) == len(decode_betas)
    decode_start_time = time.time()

    print('Generated text at epoch %d: %s ...' % (epoch, text_prompt))
    # First get greedy decoding
    greedy_text = gen_some_text_wrapper(generator_model, tokenizer, device, text_prompt, 'greedy')
    print("Greedy decoding:\n\t%s" % (greedy_text))
    # Now get several sampler decodings
    for idx in range(len(decode_seeds)):
        generated_text = gen_some_text_wrapper(generator_model, tokenizer,
                                               device,
                                               text_prompt,
                                               nongreedy_style,
                                               decode_seed=decode_seeds[idx],
                                               decode_beta=decode_betas[idx])
        print("(%s, seed=%d, beta=%.2f):\n\t%s" % (nongreedy_style, decode_seeds[idx], decode_betas[idx], generated_text))
    print('-' * 89)
    print('Epoch {:3d} | example generation time: {:5.2f}s'.format(epoch, (time.time() - decode_start_time)))
    print('-' * 89)
