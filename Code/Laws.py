import numpy as np
import matplotlib.pyplot as plt
import pingouin as pg
import pandas as pd
from fasttext import FastText



class laws():
    def __init__(self, lingua):
        if lingua == 'spanish':
            self.lingua = 'spa'
            self.word = 2
        elif lingua == 'italian':
            self.lingua = 'ita'
            self.word = 1

    def frequency(self):
        
        data =[]

        with open(f'files/filtered_lista_{self.lingua}.csv', 'r') as f:
            for element in f:
                values = element.split(',')
                word = str(values[0])
                freq = float(values[1])
                poly = float(values[2])
                cos = float(values[3])
                data.append((word,freq,poly,cos))

        #Data sorting by frequency
        sorted_data = sorted(data, key=lambda x: x[1], reverse=True)
        

        #Split the sorted data into separate lists
        Frequency = [item[1] for item in sorted_data]
        Polysemy = [item[2] for item in sorted_data]
        Cos_sim = [item[3] for item in sorted_data]


        #Calculate the sum of all values in the Frequency list
        sum_frequency = sum(Frequency)

        #Perform normalization on the Frequency values by dividing each value by the sum
        normalized_frequency = [value / sum_frequency for value in Frequency]

        import numpy as np
        from scipy import stats

        #Convert the lists to numpy arrays
        variable1 = np.array(normalized_frequency)
        variable2 = np.array(Cos_sim)
        covariate = np.array(Polysemy)

        #Display the result of correlation
        print('The correlation between Frequency and Cosine similarity is:')
        x =pg.corr(variable1, variable2).round(3)
        print(x['r'].item())

        #Create DataFrame
        df = pd.DataFrame({'Frequency': normalized_frequency , 'Cos_sim': Cos_sim, 'Polysemy': Polysemy})

        print('The partial correlation between Frequency and Cosine similarity is:')

        #Calculate partial correlation
        y = pg.partial_corr(data=df, x='Frequency', y='Cos_sim', covar='Polysemy').round(3)

        #Display the result of partial correaltion
        print(y['r'].item())

        # Create scatter plot
        plt.scatter(variable1, variable2)
        plt.xlabel('Frequency values (normalised)')
        plt.ylabel('Cosine similarity values')

        #Add trendline
        z = np.polyfit(variable1, variable2, 1)
        p = np.poly1d(z)
        plt.plot(variable1, p(variable1), "r--")

        #Calculate the correlation value
        correlation_value = x['r'].item()

        #Set the correlation value as the plot title
        plt.title(f"Correlation: {correlation_value}")

        #Display the plot
        plt.show()


    def polysemy(self):
        
        data =[]

        with open(f'files/filtered_lista_{self.lingua}.csv', 'r') as f:
            for element in f:
                values = element.split(',')
                word = str(values[0])
                freq = float(values[1])
                poly = float(values[2])
                cos = float(values[3])
                data.append((word,freq,poly,cos))

        #data sorting by frequency
        sorted_data = sorted(data, key=lambda x: x[2], reverse=True)
        
        
        #Split the sorted data into separate lists
        Frequency = [item[1] for item in sorted_data]
        Polysemy = [item[2] for item in sorted_data]
        Cos_sim = [item[3] for item in sorted_data]

        #Calculate the sum of all values in the Frequency list
        sum_frequency = sum(Frequency)
        # Perform normalization on the Frequency values by dividing each value by the sum
        normalized_frequency = [value / sum_frequency for value in Frequency]

        #Convert the lists to numpy arrays
        variable1 = np.array(Polysemy)
        variable2 = np.array(Cos_sim)
        covariate = np.array(Frequency)

        #Display the result of correlation
        x =pg.corr(variable1, variable2).round(3)
        print('The correlation between Polysemy and Cosine similarity is:')

        print(x['r'].item())

        #Create DataFrame
        df = pd.DataFrame({'Polysemy': Polysemy, 'Cos_sim': Cos_sim, 'Frequency': normalized_frequency})

        #Calculate partial correlation
        y = pg.partial_corr(data=df, x='Polysemy', y='Cos_sim', covar='Frequency').round(3)
        print('The partial correlation between Polysemy and Cosine similarity is:')

        #Display the result of partial correlation
        print(y['r'].item())


        #Create scatter plot
        plt.scatter(variable1, variable2)
        plt.xlabel('Polysemy values')
        plt.ylabel('Cosine similarity values')

        #Add trendline
        z = np.polyfit(variable1, variable2, 1)
        p = np.poly1d(z)
        plt.plot(variable1, p(variable1), "r--")

        #Calculate the correlation value
        correlation_value = x['r'].item()

        #Set the correlation value as the plot title
        plt.title(f"Correlation: {correlation_value}")

        #Display the plot
        plt.show()


    def analogy(self):

        while True:

            #Load the models and variables
            model_hist = FastText.load_model(f'Models/{self.lingua}_historical.bin')
            model_mod = FastText.load_model(f'Models/{self.lingua}_modern.bin')

            #Get the list of cognates
            self.cog_list =[]
            with open(f'files/Cog_database.csv', 'r') as f:
                for element in f:
                    values = element.split(',')
                    number = int(values[0])
                    ita = str(values[1])
                    spa = str(values[2])
                    self.cog_list.append((number,ita,spa))

            #Populate the list with the cognates of the chosen language
            wordslist = []
            for element in self.cog_list:
                print(element[self.word])
                wordslist.append(element[self.word])
            
            #Input the word
            while True:

                word = input('Choose a word from the list above: ')

                if str(word) in wordslist:
                    self.w = word
                    break
                else:
                    print('The selected word is not present in the list. Try again: ')

            #Input the value of K Nearest Neighbours
            while True:

                k = input('Choose the number of K Nearest Neighbours: ')

                if k.isdigit():
                    self.k = int(k)
                    break  # Exit the loop if a valid number is entered
                else:
                    print("Invalid input. Please enter a valid number for K.")
            
            
            c=0
            hist_nn =[]
            mod_nn = []
            
            print('Computing..')
            hist_nn.append(model_hist.get_nearest_neighbors(self.w, self.k))
            mod_nn.append(model_mod.get_nearest_neighbors(self.w, self.k))

            # Split the words and the cos in separate list
            hist_word = [[t[1] for t in lst] for lst in hist_nn]
            hist_sim = [[t[0] for t in lst] for lst in hist_nn]


            mod_word = [[t[1] for t in lst] for lst in mod_nn]
            mod_sim = [[t[0] for t in lst] for lst in mod_nn]

            hist_a=[]#hist
            mod_b=[]#mod

            #Populate the list with the NN and the correspondent Cos sim values
            for i in range(len(hist_word)):
                count = 0
                correspondence_list = []
                for j in range(len(hist_word[i])):
                    for u in range(len(hist_word[i])):
                        if hist_word[i][j] == mod_word[i][u]:
                            correspondence_list.append(hist_word[i][j])
                            hist_a.append(hist_sim[i][j])
                            mod_b.append(mod_sim[i][u])

            # if not empty, calculate the difference between the cosine similarity values
            if len(correspondence_list) > 0:
                n = 0
                summ = 0
                for index in range(len(correspondence_list)):
                    word = correspondence_list[index]
                    diff = float(hist_a[index]) - float(mod_b[index])
                    n +=1
                    summ += diff
                    print(f"{index+1} - For the word '{word}', the shift in cosine similarity is: {diff:.6f}")
                avg = summ/n
                print(f'The average difference between the K-NN is {avg:.6f}')
            

                #Plot the graph
                x = range(len(correspondence_list))

                plt.scatter(x, hist_a, color='blue', label='Historical Vector Space')
                plt.scatter(x, mod_b, color='red', label='Modern Vector Space')

                plt.xlabel('Nearest Neighbour Index')
                plt.ylabel('Cosine Value')
                plt.title(f'Vector Space Comparison for the word: "{self.w}". Avg. diff. = {avg:.6f}')
                plt.legend()

                plt.show()
                break
            else:
                print('No correspondences found!')
    
    
    def analogy2(self):
        #Load the models and variables
        model_hist = FastText.load_model(f'Models/{self.lingua}_historical.bin')
        model_mod = FastText.load_model(f'Models/{self.lingua}_modern.bin')
        occ = 0
        coherent_shift = 0
        one_nn = 0

        #Get the list of cognates
        self.cog_list = []
        with open(f'files/Cog_database.csv', 'r') as f:
            for element in f:
                values = element.split(',')
                number = int(values[0])
                ita = str(values[1])
                spa = str(values[2])
                self.cog_list.append((number, ita, spa))

        #Populate the list with the cognates of the chosen language
        wordslist = []
        for element in self.cog_list:
            wordslist.append(element[self.word])

        #Input the value of K Nearest Neighbours
        while True:
            k = input('Choose the number of K Nearest Neighbours: ')

            if k.isdigit():
                self.k = int(k)
                break  # Exit the loop if a valid number is entered
            else:
                print("Invalid input. Please enter a valid number for K.")
    
        #check for every word in the wordlist the overlapping NN
        for word in wordslist:
            self.w = word

            hist_nn = []
            mod_nn = []
            
            print('Computing...')
            hist_nn.append(model_hist.get_nearest_neighbors(self.w, self.k))
            mod_nn.append(model_mod.get_nearest_neighbors(self.w, self.k))

            # Split the words and the cos in separate list
            hist_word = [[t[1] for t in lst] for lst in hist_nn]
            hist_sim = [[t[0] for t in lst] for lst in hist_nn]

            mod_word = [[t[1] for t in lst] for lst in mod_nn]
            mod_sim = [[t[0] for t in lst] for lst in mod_nn]

            hist_a = []  # hist
            mod_b = []  # mod

            #Populate the list with the NN and the correspondent Cos sim values
            for i in range(len(hist_word)):
                count = 0
                correspondence_list = []
                for j in range(len(hist_word[i])):
                    for u in range(len(hist_word[i])):
                        if hist_word[i][j] == mod_word[i][u]:
                            correspondence_list.append(hist_word[i][j])
                            hist_a.append(hist_sim[i][j])
                            mod_b.append(mod_sim[i][u])

            # if not empty, calculate the difference between the cosine similarity values
            if len(correspondence_list) > 0:
                pos_diff_count = 0
                neg_diff_count = 0
                pos_ratio = 0
                neg_ratio = 0

                for index in range(len(correspondence_list)):
                    diff = float(hist_a[index]) - float(mod_b[index])
                    if diff > 0:
                        pos_diff_count += 1
                    elif diff < 0:
                        neg_diff_count += 1
                    pos_ratio = pos_diff_count/(pos_diff_count+neg_diff_count)
                    neg_ratio = neg_diff_count/(pos_diff_count+neg_diff_count)
            #Print computations
                print(f"Word: {word}")
                print(f"Number of NN that shift closer to the cognate: {pos_diff_count}")
                print(f"Number of NN that shift further from the cognate: {neg_diff_count}")
                print(f"Positive shift ratio: {pos_ratio}")
                print(f"Negative shift ratio: {neg_ratio}")
                print("-------------------------------")
                if pos_ratio > 0.75 and (pos_diff_count+neg_diff_count) != 1 or neg_ratio > 0.75 and (pos_diff_count+neg_diff_count) != 1:
                    coherent_shift += 1
                if (pos_diff_count+neg_diff_count) == 1:
                    one_nn +=1

            else:
                occ +=1
                print(f"No correspondences found for the word: {word}")
        #Print results
        print(f'Of {len(wordslist)} words, there are {occ} whitout any NN')
        print(f'In the words found ({(len(wordslist)-occ)}) there are {one_nn} words with only one NN.')
        print(f'In the remaining {(len(wordslist)-occ-one_nn)} words, there are {coherent_shift} with a coherent shift in the K-NN')
