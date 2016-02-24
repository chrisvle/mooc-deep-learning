
# train the model, output generated text after each iteration
#total_iterations = 0
for iteration in range(1, iteration_count):
    print()
    print('-' * 50)
    print('Iteration', total_iterations)
    model.fit(X, y, batch_size=256, nb_epoch=1)
    total_iterations += 1
    start_index = random.randint(0, len(X))
    if output_iteration_flag:
      if total_iterations % output_iteration == 0 or iteration == 1:
        print("writing model")
        model.save_weights(output_dir+"w"+str(window_len)+"i"+str(total_iterations)+".h5", overwrite=True)
        
    for diversity in [0.1, 0.2, 0.4, 0.8]:
        print()
        print('----- diversity:', diversity)

        generated = []
        sentence = X[start_index]
        word_sentence = [indices_word[wd] for wd in sentence if wd > 0]
        #generated.extend(word_sentence)
        new_str = ' '.join(word_sentence)
        print("Window size: " + str(window_len) + " and step size: " + str(step_size))
        print('----- Generating with seed: ' + new_str)
        #sys.stdout.write(word_sentence)


        for i in range(50):
            x = np.zeros((1, len(sentence)))
            for t, char in enumerate(sentence):
                x[0, t] = sentence[t]
            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds[-1], diversity) + 1 #Sample produces the index of the one hot vector, so we have to add 1 to come back to our 1-indexing
            next_word = indices_word[next_index] #have to add 1 due to this returning 0-index, but dictionary is 1-indexed

            generated.append(next_word)
            sentence = np.append(sentence, [next_index])

            #sys.stdout.write(next_word)
            #sys.stdout.flush()
        #word_sentence = [indices_word[wd] for wd in sentence]
        print(' '.join(generated))
        generated_examples[(total_iterations, diversity)] = word_sentence + generated
        #saved_generations[(total_iterations, diversity)] = ' '.join(generated)
        


"""
for iteration in range(1, 10):
    total_iterations+=1
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=256, nb_epoch=1)

    start_index = random.randint(0, len(X))

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = []
        sentence = X[start_index]
        word_sentence = [indices_word[wd] for wd in sentence]
        generated.extend(word_sentence)
        new_str = ' '.join(generated)
        print('----- Generating with seed: ' + new_str)
        #sys.stdout.write(word_sentence)

        for iteration in range(20):
            x = np.zeros((1, window_len))
            for t, char in enumerate(sentence):
                x[0, t] = sentence[t]
            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds[window_len-1], diversity)
            next_word = indices_word[next_index]

            generated.append(next_word)
            sentence = np.append(sentence[1:], [next_index])

            #sys.stdout.write(next_word)
            #sys.stdout.flush()
        word_sentence = [indices_word[wd] for wd in sentence]
        print(' '.join(generated))
        print()
"""
