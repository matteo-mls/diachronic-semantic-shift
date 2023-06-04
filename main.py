import sys
sys.path.insert(1,'Code')
from Analysis import analysis
from Laws import laws

class semantic_shift():
    def __init__(self):
        self.lingua = ''    


    def run(self):
        #Language input
        self.lingua = input('Choose the language of interest (Italian or Spanish): ')
        self.lingua = self.lingua.lower()

        while self.lingua not in ['spanish', 'italian']:
            self.lingua = input('Choice not valid. Please choose Italian or Spanish: ')
            self.lingua = self.lingua.lower()

        #Operation input
        print('Would you like to perform analysis or check for semantic laws?')
        print('Select [1] for Analysis')
        print('Select [2] for Laws')
        self.choice = input('Your choice (1-2): ')

        while self.choice not in ['1', '2']:
            print('Select [1] for Analysis')
            print('Select [2] for Laws')
            self.choice = input('Your choice (1-2): ')

        #Analysis choice
        if self.choice == '1':
            self.analysis = analysis(self.lingua)
            print('Which analysis would you like to perform?')
            print('Select [1] for RSA')
            print('Select [2] for Cosine Similarity')
            print('Select [3] for K-NN Overlap')
            self.anal = input('Your choice (1-3): ')
            while self.anal not in ['1', '2', '3']:
                print('Select [1] for RSA')
                print('Select [2] for Cosine Similarity')
                print('Select [3] for K-NN Overlap')
                self.anal = input('Your choice (1-3): ')

            if self.anal == '1':
                self.analysis.rsa()
            elif self.anal == '2':
                self.analysis.cos_sim()
            elif self.anal == '3':
                self.analysis.knn_overlap()
                pass

        #Law choice
        elif self.choice == '2':
            self.laws = laws(self.lingua)
            print('Which laws would you like to check?')
            print('Select [1] for Law of Frequency')
            print('Select [2] for Law of Polysemy')
            print('Select [3] for Law of Analogy details')
            print('Select [4] for Law of Analogy overview')
            self.trafalgar = input('Your choice (1-3): ')
            while self.trafalgar not in ['1', '2', '3','4']:
                print('Select [1] for Law of Frequency')
                print('Select [2] for Law of Polysemy')
                print('Select [3] for Law of Analogy details')
                print('Select [4] for Law of Analogy overview')
                self.trafalgar = input('Your choice (1-4): ')
            
            if self.trafalgar == '1':
                self.laws.frequency()
            if self.trafalgar == '2':
                self.laws.polysemy()
            if self.trafalgar == '3':
                self.laws.analogy()
            if self.trafalgar == '4':
                self.laws.analogy2()
             
#Run the code
a = semantic_shift()
a.run()
