# Custom-NER
Machine Learning NER Custom Model and Network Graph Visualization
# PREFACE 
	How Much Data Is Generated Every Day? 
	According to the latest estimates, 328.77 million terabytes of data are created each day.(“Created” includes data that is newly generated, captured, copied, or consumed).
	In zettabytes, that equates to 120 zettabytes per year, 10 zettabytes per month, 2.31 zettabytes per week, or 0.33 zettabytes every day.
	Our use case is: 	
	Law Enforcement Agency has drowned in piles of structured and non-structured documents of criminal cases. These criminal cases and its documents demand huge amount of time and resources to process.
	What is most important part of every document? It may be Named Entities and semantic links between them that the document contains.
	Can Machine Learning models help us to recognize Named Entities and semantic links between them?
	Yes, there is a great bunch of pre-trained models, that are able to  recognize set of defined Named Entities. But what if we need some specific entities from various branches?
	Here is our Plan of Attack: 
    • to build Named Entity Recognition system based on spaCy to process textually unstructured documents (PDFs, Doc, HTML etc.);  
    • to enrich default list of model's Named Entities  with custom types of Entities, based on needs of stakeholder (for example - "Drugs", "Weapon", "Crime" etc.);
    • to retrain model with custom types of Entities; 
    • to bring marked Named Entities into Link Analysis (Social Network Analysis) tool for visualization;
    • find and visualize links between Named Entities on the link chart;
    • to accumulate detected Named Entities into local or cloud store.
# MAIN PART
	So, as a base of NER system we take the following steps:
	1. We take spaCy model "en_core_web_lg";
	2. Then we add custom labels that are absent in standard model (in our case "DRUG" and "GANG_ORG");
 	3. We add entity_ruler which can be combined with the statistical EntityRecognizer to boost accuracy, or used on its own to implement a purely rule-based entity recognition system
	4. We add patterns into NER component of the model (as JSON files "drugs_patterns.jsonl" and "gang_patterns.jsonl")
 	5. We convert training data from JSON to spaCy format
  	6. We retrain the model with training data
   	7. We test the model with new unseen data
    	8. We build graph chart with recognized Named Entities
All above-mentioned steps contained in "Training_of_the_Custom_Model_1-Copy1.ipynb" notebook in this repo.
Front-end of the model developed in Streamlit. You can find it in Python files "upl_8.py", "custom_scorer_2.py"in 
Training data is prepared in Data Studio application with help of predictions, generated with script of "preannotated.py"
# What is left to do?
1. To implement NER coreference system.
2. To implement Entity Linker based on semantic links in sentences and the whole text.
3. To implement storing results of NER to the local or cloud-based Database.

