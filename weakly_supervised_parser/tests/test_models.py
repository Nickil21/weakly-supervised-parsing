import numpy as np
from huggingface_hub import hf_hub_url, cached_download

import nltk

nltk.download("stopwords")

from weakly_supervised_parser.inference import Predictor
from weakly_supervised_parser.tree.evaluate import calculate_F1_for_spans, tree_to_spans
from weakly_supervised_parser.model.trainer import InsideOutsideStringClassifier


validation_sentences = [
    """Skipper 's Inc. Bellevue Wash. said it signed a definitive merger agreement for a National Pizza Corp. unit to
       acquire the 90.6 % of Skipper 's Inc. it does n't own for 11.50 a share or about 28.1 million""",
    """NP Acquisition Co. a National Pizza unit plans to begin a tender offer for Skipper 's on Friday conditioned on at 
       least two-thirds of Skipper 's shares being tendered""",
    """Pittsburg Kan.-based National Pizza said the transaction will be financed under its revolving credit agreement""",
    """In national over-the-counter trading Skipper 's shares rose 50 cents to 11""",
    """Skipper 's said the merger will help finance remodeling and future growth""",
    """Skipper 's previously turned down a 10-a-share proposal from National Pizza and Pizza Hut Inc. questioned whether 
       the purchase would violate National Pizza 's franchise agreements""",
    """National Pizza said it settled its dispute with Pizza Hut allowing it to make the purchase""",
    """Also Skipper 's results began to turn around permitting a higher offer National Pizza said""",
    """For the 12 weeks ended Sept. 3 Skipper 's had net income of 361,000 or 13 cents a share compared with a net loss a year earlier""",
    """Revenue was 19.9 million""",
]

validation_gold_trees = [
    """(S (NP-SBJ (NP (NP (NNP Skipper) (POS 's)) (NNP Inc.)) (NP-LOC (NP (NNP Bellevue)) (NP (NNP Wash.)))) 
       (VP (VBD said) (SBAR (S (NP-SBJ (PRP it)) (VP (VBD signed) (NP (DT a) (JJ definitive) (NN merger) (NN agreement) 
       (SBAR (IN for) (S (NP-SBJ (DT a) (NNP National) (NNP Pizza) (NNP Corp.) (NN unit)) (VP (TO to) (VP (VB acquire) (NP (NP (DT the) 
       (CD 90.6) (NN %)) (PP (IN of) (NP (NP (NNP Skipper) (POS 's)) (NNP Inc.))) (SBAR (S (NP-SBJ (PRP it)) (VP (VBZ does) (RB n't) 
       (VP (VB own)))))) (PP-CLR (IN for) (NP (NP (NP (CD 11.50)) (NP-ADV (DT a) (NN share))) (CC or) (NP (QP (RB about) (CD 28.1) 
       (CD million)))))))))))))))""",
    """(S (NP-SBJ-1 (NP (NNP NP) (NNP Acquisition) (NNP Co.)) (NP (DT a) (NNP National) (NNP Pizza) (NN unit))) (VP (VBZ plans) (S (VP (TO to) 
       (VP (VB begin) (NP (NP (DT a) (NN tender) (NN offer)) (PP (IN for) (NP (NNP Skipper) (POS 's)))) (PP-TMP (IN on) (NP (NNP Friday))) 
       (PP (VBN conditioned) (PP (IN on) (NP (NP (QP (IN at) (JJS least) (NNS two-thirds))) (PP (IN of) (NP (NP (NP (NNP Skipper) (POS 's)) 
       (NNS shares)) (VP (VBG being) (VP (VBN tendered)))))))))))))""",
    """(S (NP-SBJ (ADJP (NNP Pittsburg) (JJ Kan.-based)) (NNP National) (NNP Pizza)) (VP (VBD said) (SBAR (S (NP-SBJ-1 (DT the) (NN transaction)) 
       (VP (MD will) (VP (VB be) (VP (VBN financed) (PP-LOC (IN under) (NP (PRP$ its) (JJ revolving) (NN credit) (NN agreement))))))))))""",
    """(S (PP-LOC (IN In) (NP (JJ national) (JJ over-the-counter) (NN trading))) (NP-SBJ (NP (NNP Skipper) (POS 's)) (NNS shares)) (VP (VBD rose) 
       (NP-EXT (CD 50) (NNS cents)) (PP-DIR (TO to) (NP (CD 11)))))""",
    """(S (NP-SBJ (NP (NNP Skipper) (POS 's))) (VP (VBD said) (SBAR (S (NP-SBJ-1 (DT the) (NN merger)) (VP (MD will) (VP (VB help) (S (VP (VB finance) 
       (NP (NP (NN remodeling)) (CC and) (NP (JJ future) (NN growth)))))))))))""",
    """(S (S (NP-SBJ (NP (NNP Skipper) (POS 's))) (VP (ADVP-TMP (RB previously)) (VBD turned) (PRT (RP down)) (NP (NP (DT a) (ADJP (JJ 10-a-share)) 
       (NN proposal)) (PP (IN from) (NP (NNP National) (NNP Pizza)))))) (CC and) (S (NP-SBJ (NNP Pizza) (NNP Hut) (NNP Inc.)) (VP (VBD questioned) 
       (SBAR (IN whether) (S (NP-SBJ (DT the) (NN purchase)) (VP (MD would) (VP (VB violate) (NP (NP (NNP National) (NNP Pizza) (POS 's)) (NN franchise) 
       (NNS agreements)))))))))""",
    """(S (NP-SBJ (NNP National) (NNP Pizza)) (VP (VBD said) (SBAR (S (NP-SBJ (PRP it)) (VP (VBD settled) (NP (NP (PRP$ its) (NN dispute)) 
       (PP (IN with) (NP (NNP Pizza) (NNP Hut)))) (S-ADV (VP (VBG allowing) (S (NP-SBJ (PRP it)) (VP (TO to) (VP (VB make) (NP (DT the)
       (NN purchase))))))))))))""",
    """(S (S-TPC-2 (ADVP (RB Also)) (NP-SBJ-1 (NP (NNP Skipper) (POS 's)) (NNS results)) (VP (VBD began) (S (VP (TO to) (VP (VB turn) 
       (PRT (RP around)) (S-ADV (VP (VBG permitting) (NP (DT a) (JJR higher) (NN offer))))))))) (NP-SBJ (NNP National) (NNP Pizza)) (VP (VBD said)))""",
    """(S (PP-TMP (IN For) (NP (NP (DT the) (CD 12) (NNS weeks)) (VP (VBN ended) (NP-TMP-CLR (NNP Sept.) (CD 3))))) (NP-SBJ (NP (NNP Skipper)
       (POS 's))) (VP (VBD had) (NP (NP (JJ net) (NN income)) (PP (IN of) (NP (NP (CD 361,000)) (CC or) (NP (NP (CD 13) (NNS cents)) (NP-ADV (DT a)
       (NN share)))))) (PP (VBN compared) (PP (IN with) (NP (NP (DT a) (JJ net) (NN loss)) (ADVP-TMP (NP (DT a) (NN year)) (RBR earlier)))))))""",
    """(S (NP-SBJ (NN Revenue)) (VP (VBD was) (NP-PRD (QP (CD 19.9) (CD million)))))""",
]


def test_inside_model():
    inside_model = InsideOutsideStringClassifier(model_name_or_path="roberta-base", max_seq_length=256)
    fetch_url_inside_model = hf_hub_url(repo_id="nickil/weakly-supervised-parsing", filename="inside_model.onnx", revision="main")
    inside_model.load_model(pre_trained_model_path=cached_download(fetch_url_inside_model))
    sentences_f1 = []
    for validation_sentence, validation_gold_tree in zip(validation_sentences, validation_gold_trees):
        best_parse = Predictor(sentence=validation_sentence).obtain_best_parse(
            predict_type="inside", model=inside_model, scale_axis=None, predict_batch_size=512
        )
        sentence_f1 = calculate_F1_for_spans(tree_to_spans(validation_gold_tree), tree_to_spans(best_parse))
        sentences_f1.append(sentence_f1)
    assert np.mean(sentences_f1) > 50
