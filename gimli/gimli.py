import csv,time, math,random

from constants import *
from utilities import *

class Gimli:
    def __init__(self,source_csv,ngram_max=2,filter_threshold=1,
            silent=False,validate=False,validation_ratio=0.8,balanced=0):
        self.ngram_max=ngram_max
        self.filter_threshold=filter_threshold
        self.validation_ratio=validation_ratio
        self.balanced=balanced
        self.silent=silent

        if not silent:
            print_color("Loading CSV '%s'."%source_csv,COLORS.GREEN)
        self.set_source_data(source_csv)
        if not silent:
            print_color("Processing data.",COLORS.GREEN)
        self.process_training_data()
        if not silent:
            print_color("Scoring ngrams.",COLORS.GREEN)
        self.score_ngrams()
        if validate and self.validation_ratio<1:
            self.validate()

    def make_predictions_csv(self,target):
        lines=[]
        with open(target,newline="") as f:
            reader=csv.reader(f)
            is_header=True
            for row in reader:
                if is_header:
                    is_header=False
                    continue
                if not row or len(row)!=2 or not row[0]:
                    continue
                id,text=row
                prediction=self.get_prediction(text)
                lines.append((str(id),str(prediction)))

        with open(add_filename_prefix_to_path("predictions-",target),"w") as f:
            f.write("Id,Prediction\n")
            for line in lines:
                f.write(",".join(line)+"\n")

    def get_prediction(self,text):
        confidence=self.get_prediction_tuple(text)
        return self.get_prediction_from_tuple(confidence)

    def get_prediction_from_tuple(self,confidence):
        return confidence.index(max(confidence))

    def get_prediction_tuple(self,text):
        ngrams=get_cumulative_ngrams(text,self.ngram_max)
        confidence=[0 for i in range(OPTION_COUNT)]

        for ngram in ngrams:
            if ngram not in self.ngram_scores or count_tokens(ngram)>0:
                continue
            scores=self.ngram_scores[ngram]
            confidence=[confidence[i]+scores[i]**3 for i in range(OPTION_COUNT)]

        total=sum(confidence)
        total=total if total else 1
        confidence=[i/total for i in confidence]
        return confidence

    def set_source_data(self,source_csv):
        self.data=[]

        with open(source_csv,newline="") as f:
            reader=csv.reader(f)
            is_header=True
            for row in reader:
                if not row or len(row)!=3 or not row[0]:
                    continue
                if is_header:
                    is_header=False
                    continue
                id,text,category=row
                item=(int(id),text,int(category))
                self.data.append(item)

        random.shuffle(self.data)
        self.training_data=self.data[:int(len(self.data)*self.validation_ratio)]
        self.validation_data=self.data[int(len(self.data)*self.validation_ratio):]

        if self.balanced:
            self.training_data=self.balance_data(self.training_data)

    def balance_data(self,data):
        mins=[0 for i in range(OPTION_COUNT)]
        for id,text,category in data:
            mins[category]+=1
        mincat=min(mins)

        if not self.silent:
            print_color("Balancing data. Minimum count=%s"%mincat,COLORS.YELLOW)
        new_data=[]
        counter=[0 for i in range(OPTION_COUNT)]
        for item in data:
            a,b,category=item
            if counter[category]<mincat:
                new_data.append(item)
            counter[category]+=1
        return new_data

    def process_training_data(self):
        #where keys are ngrams, values are (0,0,0,0) for number of times they occur for each category
        self.ngram_count={}
        #each number is how many ngrams were processed for that category in total
        self.ngram_total=[0 for i in range(OPTION_COUNT)]

        counter=0
        timer=Timer(10)
        for id,text,category in self.training_data:
            counter+=1
            timer.tick("Processing item %s/%s"%(counter,len(self.training_data)))

            grams=get_cumulative_ngrams(text,self.ngram_max)

            for gram in grams:
                if gram not in self.ngram_count:
                    self.ngram_count[gram]=[0 for i in range(OPTION_COUNT)]
                count=self.ngram_count[gram]
                count[category]+=1
                self.ngram_count[gram]=count

        #filter out ngrams with only 'filter threshold' number of occurences
        if self.filter_threshold:
            self.ngram_count={key:self.ngram_count[key] for key in self.ngram_count if sum(self.ngram_count[key])>self.filter_threshold}

        for category in range(OPTION_COUNT):
            self.ngram_total[category]=sum([self.ngram_count[ngram][category] for ngram in self.ngram_count])

    def score_ngrams(self):
        #where keys are ngrams, values are (0,0,0,0) which add up to 1, meaning belief in each category
        self.ngram_scores={}
        normalized=[self.ngram_total[i]/max(self.ngram_total) for i in range(OPTION_COUNT)]

        i=0
        timer=Timer(10)
        for ngram in self.ngram_count:
            i+=1
            timer.tick("Scoring item %s/%s"%(i,len(self.ngram_count)))

            count=self.ngram_count[ngram]
            adjusted_count=[0 if not normalized[i] else count[i]/normalized[i] for i in range(OPTION_COUNT)]
            total=sum(adjusted_count)
            scores=[adjusted_count[i]/total for i in range(OPTION_COUNT)]
            self.ngram_scores[ngram]=scores

    def validate(self):
        correct=0
        matrix=[[0 for i in range(OPTION_COUNT)] for j in range(OPTION_COUNT)]
        confidence_total=[0 for i in range(OPTION_COUNT)]
        prediction_counter=[0 for i in range(OPTION_COUNT)]

        for id,text,category in self.validation_data:
            confidence=self.get_prediction_tuple(text)
            prediction=self.get_prediction_from_tuple(confidence)

            confidence_total[prediction]+=confidence[prediction]
            prediction_counter[prediction]+=1

            if category==prediction:
                correct+=1
            matrix[category][prediction]+=1

        percent=round(100*correct/len(self.validation_data),2)
        print_color("Validation: %s/%s correct. %s%%"%(correct,len(self.validation_data),percent),COLORS.PURPLE)

        print_color("Confidence per category:",COLORS.PURPLE)
        for i in range(OPTION_COUNT):
            percent=round(100*confidence_total[i]/prediction_counter[i],2)
            print_color("%s: %s%%"%(i,percent),COLORS.PURPLE,end="")
            print_color(" | ",COLORS.YELLOW, end="")
        print("")

        print_color("Prediction matrix. X=reality  Y=prediction",COLORS.MAGENTA)
        self.show_matrix(matrix)

    def show_matrix(self,matrix):
        bar="-"*34
        print_color(bar,COLORS.YELLOW,end="")
        for j in range(OPTION_COUNT):
            print_color("\n | ",COLORS.YELLOW,end="")
            for i in range(OPTION_COUNT):
                print_color(str(matrix[i][j]).rjust(5),COLORS.MAGENTA,end="")
                print_color(" | ",COLORS.YELLOW,end="")
            percent=100*matrix[j][j]/sum([matrix[i][j] for i in range(OPTION_COUNT)])
            print_color(" %s%%"%round(percent,2),COLORS.PURPLE,end="")

        print_color("\n"+bar,COLORS.YELLOW)

        print(" ",end="")
        for i in range(OPTION_COUNT):
            percent=100*matrix[i][i]/sum([matrix[i][j] for j in range(OPTION_COUNT)])
            text=(" %s%%"%round(percent,1)).rjust(8)
            print_color(text,COLORS.PURPLE,end="")
        print("")












