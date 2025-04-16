How to use:
    1. Install packages (spacy + newspaper3k)
        * I used conda, run:
            a. conda install -c conda-forge newspaper3k
            b. conda install -c conda-forge spacy
    
    2. Install the model to find noun phrases:
        run python -m spacy download en_core_web_sm

    3. Run python NewsSearch.py input_file article_count
        i.e: python NewsSearch.py News_Category_Dataset_v3.json 10000
        a. if no article_count count was inserted, the every article_count
            will be processed. For all of News_Category_Dataset_v3, this
            will take ~20 hours
        b. Output:
            i. Parsed_News.json: contains all the single + np TFs
            ii. single_word_idf.json and np_idf.json contains the IDFs