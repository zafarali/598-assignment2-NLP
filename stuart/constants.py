
from nltk.corpus import stopwords

ENGLISH_STOPWORDS=stopwords.words('english')

OPTION_COUNT=4

TOKENS={"?":"question",".":"period",",":"comma",";":"semicolon",":":"colon","-":"dash","!":"exclamation"}
TOKENS={key:" _"+TOKENS[key]+" " for key in TOKENS}

class COLORS:
    RED=41
    GREEN=42
    ORANGE=43
    BLUE=44
    PURPLE=45
    AQUA=46
    SILVER=47
    GREY=100
    PINK=101
    LIME=102
    YELLOW=103
    SKY=104
    MAGENTA=105
    TURQUOISE=106
    WHITE=107
