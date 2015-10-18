import csv,time,math,random

from constants import *
from utilities import *

class TextAnalyzer:
    def __init__(self,source_csv,silent=False,validation_ratio=0.8,interval=20,balanced=False):
        self.validation_ratio=validation_ratio
        self.silent=silent
        self.interval=interval
        self.has_processed=False

        if not silent:
            print_color("Loading CSV '%s'."%source_csv,COLORS.GREEN)
        self.set_source_data(source_csv)
        if balanced:
            self.balance_data()

    def make_predictions_csv(self,target):
        if not self.has_processed:
            raise ValueError("ABORT. TextAnalyzer never processed.")
        lines=[]
        timer=Timer(self.interval)
        counter=0
        with open(target,"r") as f:
            line_count=len(f.readlines())

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
                counter+=1
                timer.tick("Making prediction %s/%s."%(counter,line_count))
                prediction=self.get_prediction(text)
                lines.append((str(id),str(prediction)))

        with open(add_filename_prefix_to_path("predictions-",target),"w") as f:
            f.write("Id,Prediction\n")
            for line in lines:
                f.write(",".join(line)+"\n")

    def get_prediction(self,text):
        if not self.has_processed:
            raise ValueError("ABORT. TextAnalyzer never processed.")
        confidence=self.get_prediction_tuple(text)
        return self.get_prediction_from_tuple(confidence)

    def get_prediction_from_tuple(self,confidence):
        if not self.has_processed:
            raise ValueError("ABORT. TextAnalyzer never processed.")
        return confidence.index(max(confidence))

    def get_prediction_tuple(self,text):
        raise NotImplemented

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

    def balance_data(self):
        cat_total=[0 for i in range(OPTION_COUNT)]
        for id,text,category in self.training_data:
            cat_total[category]+=1
        max_items=min(cat_total)

        cat_count=[0 for i in range(OPTION_COUNT)]
        new_td=[]
        for item in self.training_data:
            a,b,category=item
            if cat_count[category]<max_items:
                cat_count[category]+=1
                new_td.append(item)
        self.training_data=new_td

    def validate(self):
        if not self.has_processed:
            raise ValueError("ABORT. TextAnalyzer never processed.")
        if len(self.validation_data)==0:
            print_color("Not validating - no validation data. Maybe validation ratio is 1? (100% used for training)",COLORS.RED)
            return
        correct=0
        matrix=[[0 for i in range(OPTION_COUNT)] for j in range(OPTION_COUNT)]
        confidence_total=[0 for i in range(OPTION_COUNT)]
        prediction_counter=[0 for i in range(OPTION_COUNT)]

        timer=Timer(self.interval)
        counter=0

        for id,text,category in self.validation_data:
            counter+=1
            timer.tick("Validating item %s/%s"%(counter,len(self.validation_data)))
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
        if not self.has_processed:
            raise ValueError("ABORT. TextAnalyzer never processed.")
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












