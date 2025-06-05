""" 
This is a semi-automated extraction of topics from the free text data associated with the. This is used to 
    determine the level of dependence in the dataset.

@author
"""
from re import sub
from numpy import nan
from pandas import read_csv
from nltk import tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

def to_topics(text, exclude={}, tags=['NN'], delim=',', demo=False):
    """
    * Extract a comma separated list of lemmatised nouns (NN, or other specified 
    *    POS tag) from a text string
    """
    # return nothing if there is nothing to process
    if text in [nan, '', False]:
        return ""
    
    # split into sentences then tokens - split at any dodgy characters too
    tokens = [tokenize.word_tokenize(sub(r'[^a-zA-Z0-9]', ' ', s)) for s in tokenize.sent_tokenize(text)]

    # part of speech tagging
    pos_tagged_tokens = [pos_tag(t) for t in tokens]
    pos_tagged_tokens = [token for sent in pos_tagged_tokens for token in sent]
    if demo:
        print(pos_tagged_tokens)
    
    # get lemmatized nouns into a set (no duplicates)
    lemmatizer = WordNetLemmatizer()
    terms = {lemmatizer.lemmatize(token).lower() for(token, pos) in pos_tagged_tokens if pos[:2]in tags and len(token) > 2}

    # enforce excluded terms
    if demo:
        print(f"pre-exclusion: {len(terms)}")
    terms = [t for t in terms if t not in exclude]
    if demo:
        print(f"post-exclusion: {len(terms)}")
        
    # concat into separated string and return
    return delim.join(terms)


def get_all_topics(topics_list):
    """ Convert series of comma-separated topics into a list of unique topics"""
    return { t for topics in topics_list for t in topics.split(",") if t != ''}

if __name__ == "__main__":
    # load dataframe and filter to questions of interest
    df = read_csv('data/map-me_answers_23-10-2023_12-39.csv')
    df = df[df.id_subquestion.isin([27468.0, 27469.0])]
    
    # extract terms
    df['terms'] = df.answer.apply(to_topics, args=({'tree','space','area','top','place','landscape','return','increase','size',
                                                     'damage','benefit','way','size','part','condition','west','grasmere'
                                                     'bit','prefer','terrain','visitor','research','honest','deforestation','please'
                                                     'places','outwards','pattern','introduction','specie','bottom','forest',
                                                     'monoculture','ground','motivation','combat','impact','lot','greenery','bio',
                                                     'aspect','amount','tress','wouldnt','access','bare','soil','right','left','site',
                                                     'map','stone','wind', 'season','soil','specie','root','hilltop','area','cause',
                                                     'uprooting','winter','placement','autumn','wind','plant','space','creation',
                                                     'mix','cover','tree','animal','range','topography','cooler','weather','development', 
                                                     'spoil', 'conifer', 'elevation', 'necessity', 'charcoal', 'reduction', 'house', 'density', 
                                                     'man', 'howevr', 'lakeland', 'expand', 'line', 'appalaichain', 'manmade', 'type', 
                                                     'stalwart', 'disease', 'compromise', 'level', 'legibility', 'face', 'closure', 'woodland', 
                                                     'flora', 'hay', 'preference', 'wood', 'question', 'balance', 'this', 'likelihood', 'natural', 
                                                     'unsure', 'planting', 'forestation', 'bit', 'world', 'also', 'rescaping', 'more', 
                                                     'risk', 'base', 'product', 'please', 'allan', 'anywhere', 'land', 'rockiest', 'could', 
                                                     'high', 'thrive', 'see', 'veiw', 'understanding', 'variety', 'them', 'ruggedness', 
                                                     'treeless', 'hazel', 'tops', 'aftermath', 'month', 'shame', 'craft', 'succession', 'link', 
                                                     'preserve', 'country', 'character', 'point', 'growth', 'spraying', 'scenery', 'walker', 
                                                     'activity', 'swathe', 'couple', 'bank', 'overgrowth', 'case', 'enough', 'planning', 
                                                     'state', 'effect', 'scape', 'spot', 'protection', 'result', 'treeplanting', 'value', 
                                                     'places', 'issue', 'feel', 'struggle', 'beautiful', 'footprint', 'silage', 'hill', 
                                                     'preveniotn', 'dilemma', 'home', 'fields', 'production', 'tranquility', 'traffic', 'idea', 
                                                     'lung', 'view', 'cant', 'one', 'exception', 'glade', 'claim', 'people', 'moment', 'middle', 
                                                     'common', 'the', 'storm', 'difficult', 'alot', 'coverage', 'erosion', 'treesbest', 'carbon', 
                                                     'idealist', 'thatbecause', 'impression', 'hardwood', 'clearer', 'areas', 'chance', 'eon', 
                                                     'decision', 'produce', 'viability', 'life', 'mitigatiom', 'lack', 'help', 'body', 'today', 
                                                     'broadleaf', 'topsoil', 'position', 'trees', 'negative', 'reforestration', 'style', 'mind', 
                                                     'sense', 'environment', 'and', 'rain', 'squirrel', 'shape', 'purpose', 'downstream', 'danger', 
                                                     'walk', 'husbandry', 'mountain', 'science', 'altitude', 'planted', 'none', 'uphill', 'capture', 
                                                     'beavers', 'management', 'income', 'etc', 'system', 'lowland', 'mitigation', 'navigation', 
                                                     'vegetation', 'trapping', 'field', 'desmond', 'summit', 'stop', 'valley', 'transition', 'fan', 
                                                     'mountainside', 'region', 'watercourse', 'plantation', 'input', 'future', 'change', 'want', 
                                                     'course', 'someone', 'town', 'opinion', 'better', 'hedgerow', 'surrounding', 'dense', 'have', 
                                                     'strip', 'extent', 'rock', 'distance', 'concern', 'pine', 'scots', 'amenity', 'let', 'fell', 
                                                     'support', 'tourism', 'car', 'example', 'number', 'resident', 'concentration', 'facility', 
                                                     'think', 'side', 'fellside', 'hillside', 'treeline', 'flatter', 'water', 'quality', 'horizon', 
                                                     'parkland', 'gap', 'grassland', 'outlook', 'forestry', 'thing', 'guess', 'slope', 'pike', 
                                                     'planet', 'break', 'infill', 'footpath', 'restoration', 'flow', 'resource', 'organism', 'use', 
                                                     'possibility', 'fern', 'need', 'hills', 'answer', 'structure', 'additional', 'edge', 'denser', 
                                                     'happens', 'frequency', 'peak', 'standard', 'livelihood', 'fill', 'some', 'contrast', 'reason', 
                                                     'district', 'grasmere', 'montane', 'estate', 'beauty', 'gully', 'dioxide','banks','don', 
                                                     },))
    
    # test extraction to set
    print(f"\ntopics: {get_all_topics(df.terms.to_list())}\n")

    # export CSV file
    df.to_csv('./data/answers_with_terms.csv')
    print("done.")