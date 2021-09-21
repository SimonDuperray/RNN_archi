from Transformation import CustomTransformation
import random, csv

class Merger:

   def __init__(self):
      self.input_ = None
      self.output_ = None
      self.percentage = 0
      self.fieldnames = ['in', 'out']
      self.rows = []

   def contains_letter(self, word):
      return word.lower().islower()

   def preprocessing(self, li):
      striped = [i.strip() for i in li]
      for i in range(len(striped)):
         if not self.contains_letter(li[i]):
            del striped[i]
      return [item for item in striped if item[:8]=='<authors']

   def replace_quotes(self, li):
      for i in range(len(li)):
         if '"' in li[i]:
            li[i] = li[i].replace('"', "'")
      return li

   def add_spaces_str(self, word):
      if '"' in word:
         word = word.replace('"', ' " ')
      if "'" in word:
         word = word.replace("'", " ' ")
      return word

   def add_spaces_list(self, li):
      for i in range(len(li)):
         if "'" in li[i]:
            li[i] = li[i].replace("'", " ' ")
         if '"' in li[i]:
            li[i] = li[i].replace('"', ' " ')
      return li

   def read_bibtex_file(self, filename):
      with open(filename, "r") as file:
         return [line.strip('\n') for line in file]

   def get_percent(self):
      return self.percentage

   def run(self):
      # open files
      correct_in = self.read_bibtex_file('./datasets/bibtex/correct.bibtex')
      incorrect_a_in = self.read_bibtex_file("./datasets/bibtex/not_correct_a.bibtex")
      incorrect_b_in = self.read_bibtex_file('./datasets/bibtex/not_correct_b.bibtex')

      # preprocess data
      correct_in = self.preprocessing(correct_in)
      incorrect_a_in = self.preprocessing(incorrect_a_in)
      incorrect_b_in = self.preprocessing(incorrect_b_in)

      # replace" by '
      correct_in = self.replace_quotes(correct_in)
      incorrect_a_in = self.replace_quotes(incorrect_a_in)
      incorrect_b_in = self.replace_quotes(incorrect_b_in)

      # apply custom transformation
      transformer = CustomTransformation()
      correct_out = transformer.transform(correct_in)
      incorrect_a_out = transformer.transform(incorrect_a_in)
      incorrect_b_out = transformer.transform(incorrect_b_in)

      # delete shift
      shift = abs(len(incorrect_a_out)-len(incorrect_b_out))
      biggest = [incorrect_a_in, incorrect_a_out] if len(incorrect_a_out)>len(incorrect_b_out) else [incorrect_b_in, incorrect_b_out]

      for i in range(shift):
         for list in biggest:
            del list[-1]
      
      # create final lists
      self.input_ = correct_in + incorrect_a_in[:len(incorrect_a_in)]
      self.output_ = correct_out + incorrect_b_out[:len(incorrect_b_out)]

      # shuffle lists
      random.Random(4).shuffle(self.input_)
      random.Random(4).shuffle(self.output_)

      incorr, tt = 0, 0
      r = []
      if len(self.input_)==len(self.output_):
         for i in range(len(self.input_)):
            if self.input_[i][17:-3]!=self.output_[i][8:-1]:
               incorr+=1
            tt+=1
            r.append(
               {
                  "in": self.add_spaces_str(self.input_[i]),
                  "out": self.add_spaces_str(self.output_[i])
               }
            )

      self.percentage = round(100*incorr/tt, 1)
      self.rows = r

      print(f"Preprocessing successfully operated !\n>>> Percentage of true negatives: {self.percentage}%")

      self.input_ = self.add_spaces_list(self.input_)
      self.output_ = self.add_spaces_list(self.output_)

      # save file
      filename = 'dataset.csv'
      path = "./datasets/"
      with open(path+filename, 'w', encoding='utf-8', newline='') as outfile:
         writer = csv.DictWriter(outfile, fieldnames=self.fieldnames)
         writer.writeheader()
         writer.writerows(self.rows)

      # log message
      print(f"{filename} created !")