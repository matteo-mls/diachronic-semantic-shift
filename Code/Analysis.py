import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from fasttext import FastText
from scipy.spatial.distance import squareform, pdist
from scipy.linalg import orthogonal_procrustes

class analysis():
    #Initialize the models based on the chosen language and unpack the cognate database
    def __init__(self, lingua):
        self.k = 0
        if lingua == 'spanish':
            lingua = 'spa'
            self.word = 2
        elif lingua == 'italian':
            lingua = 'ita'
            self.word = 1
        self.model_hist = FastText.load_model(f'Models/{lingua}_historical.bin')
        self.model_mod = FastText.load_model(f'Models/{lingua}_modern.bin')
        self.cog_list =[]
        with open(f'files/Cog_database.csv', 'r') as f:
            for element in f:
                values = element.split(',')
                number = int(values[0])
                ita = str(values[1])
                spa = str(values[2])
                self.cog_list.append((number,ita,spa))

    def rsa(self):
        Hist_cog = []
        Mod_cog = []

        #Get the word vector for each cognate
        for word in self.cog_list:
            hist_vector = self.model_hist.get_word_vector(word[self.word])
            mod_vector = self.model_mod.get_word_vector(word[self.word])
            Hist_cog.append(hist_vector)
            Mod_cog.append(mod_vector)

        #Convert the lists to numpy arrays
        Hist_cog = np.array(Hist_cog)
        Mod_cog = np.array(Mod_cog)

        #Compute the dissimilarity between vectors in each list
        dissimilarity_hist = pdist(Hist_cog, metric='euclidean')
        dissimilarity_mod = pdist(Mod_cog, metric='euclidean')

        #Convert the dissimilarity matrices to squareform matrices
        dissimilarity_matrix_hist = squareform(dissimilarity_hist)
        dissimilarity_matrix_mod = squareform(dissimilarity_mod)

        #Compute the correlation between the two dissimilarity matrices
        correlation_matrix = np.corrcoef(dissimilarity_matrix_hist.flatten(), dissimilarity_matrix_mod.flatten())[0, 1]
        print("RSA correlation:", correlation_matrix)

        #Plot the two matrices
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

        
        sns.heatmap(dissimilarity_matrix_hist, cmap='plasma', annot=False, square=True, ax=axs[0])
        axs[0].set_title(f'Historical data')

     
        sns.heatmap(dissimilarity_matrix_mod, cmap='plasma', annot=False, square=True, ax=axs[1])
        axs[1].set_title(f'Modern data')

        
        fig.text(0.5, 0.05, "RSA correlation: {:.4f}".format(correlation_matrix), ha='center', fontsize=18)

    
        plt.show()


    def cos_sim(self):
        
        aligned_hist_vec = {}
        modern_vec = {}
        
        print('Creating the list of commond words..')
        #Get the intersection of the vocabularies
        words1 = set(self.model_hist.get_words())
        words2 = set(self.model_mod.get_words())
        common_words = list(words1 & words2)
        
        print('Obtaining the vectors of the commond words..')

        #Get the vectors for the common words in each model
        vectors1 = np.vstack([self.model_hist.get_word_vector(w) for w in common_words])
        vectors2 = np.vstack([self.model_mod.get_word_vector(w) for w in common_words])
        #Get the vectors for the non-aligned vector space
        vectors_a = np.vstack([self.model_hist.get_word_vector(w) for w in words1])
        vectors_b = np.vstack([self.model_mod.get_word_vector(w) for w in words2])

        print('Aligning the two vector spaces..')

        # Align the two vector spaces using Procrustes analysis
        mtx1, mtx2 = orthogonal_procrustes(vectors1, vectors2)
        aligned_vectors1 = vectors1.dot(mtx1)  # Matrix mtx1 is used to transform the historical vectors in vectors1 to the aligned historical vectors in aligned_vectors1.
      
        #Populate the list with the historical aligned vectors
        for i, w in enumerate(common_words):
            aligned_hist_vec[w] = aligned_vectors1[i]
        #Populate the list with the modern vector
        for i, w in enumerate(words2):
            modern_vec[w] = vectors_b[i]

        print('Computing Cosine Similarity..')
        c = 0
        total_sim = 0.0
        missing_words = []
        lista_cosine = []

        #Compute cosine similarity for each word in cog_list
        for word in self.cog_list:
            if word[self.word] in aligned_hist_vec and word[self.word] in modern_vec:
                sim = np.dot(aligned_hist_vec[word[self.word]], modern_vec[word[self.word]]) / (np.linalg.norm(aligned_hist_vec[word[self.word]]) * np.linalg.norm(modern_vec[word[self.word]]))
                total_sim += sim
                c += 1
                print(f"{c} - Cosine similarity for word {word[self.word]}: {sim:.4f}")
                lista_cosine.append((c, word[self.word], sim))
            else:
                #if word is missing print that is missing and
                missing_words.append(word[self.word])
                print(f"Word {word[self.word]} not found in the vectors.")
                sim = 0
                c += 1
                lista_cosine.append((c, word[self.word], sim))

        #Sort by cosine value and print the sorted list
        sorted_lista_cosine = sorted(lista_cosine, key=lambda c: c[2], reverse=True)

        for a in sorted_lista_cosine:
            print(a)

        #Print the results
        if c > 0:
            avg_sim = total_sim / c
            print(f"Average cosine similarity is {avg_sim:.4f} ({c} cognates found).")
            avg_sim_no_null = total_sim / (c - len(missing_words))
            print(f"Average cosine similarity (without not found) is {avg_sim_no_null:.4f} ({c - len(missing_words)} cognates found).")
        else:
            print("No cognates found in the vectors.")

        print(f"{len(missing_words)} cognates not found in the vectors.")
        print(f"These cognates are: {missing_words}")


    def knn_overlap(self):

        #input the number of K - Nearest Neighbours
        while True:
            k = input('Choose the number of K Nearest Neighbours: ')

            if k.isdigit():
                self.k = int(k)
                break  # Exit the loop if a valid number is entered
            else:
                print("Invalid input. Please enter a valid number for K.")

        
        #Populate the list based on the K NN of each cognate word 
        hist_nn =[]
        mod_nn = []
        print('Getting the K Nearest Neighbour..')
        for word in self.cog_list:
            hist_nn.append(self.model_hist.get_nearest_neighbors(word[self.word], self.k))
            mod_nn.append(self.model_mod.get_nearest_neighbors(word[self.word], self.k))
        

        #Split the Nearest Neighbours in two lists
        hist_result = [[t[1] for t in lst] for lst in hist_nn]
        mod_result = [[t[1] for t in lst] for lst in mod_nn]

        tot_count = 0
        correspondences = []

        print('Calculating the overlap..')
        #Check the value of the overlap between the two lists
        for i in range(len(hist_result)):
            count = 0
            correspondence_list = []
            for j in range(len(hist_result[i])):
                for u in range(len(hist_result[i])):
                    if hist_result[i][j] == mod_result[i][u]:
                        correspondence_list.append(hist_result[i][j])
                        count += 1
            tot_count += count
            correspondences.append({"word": self.cog_list[i], "count": count, "correspondence_list": correspondence_list})

        #Sort the correspondences by count in descending order
        sorted_correspondences = sorted(correspondences, key=lambda x: x["count"], reverse=True)

        #Print the correspondences in order of count
        
        for i, correspondence in enumerate(sorted_correspondences):
            print(f"{i+1} - The number of correspondence nearest neighbours for the word /{correspondence['word'][self.word]}/ is {correspondence['count']}: {correspondence['correspondence_list']}")

        print(f"Total number of correspondences: {tot_count} out of {(i+1)*self.k}")
