from Wope import Copoet
ai_poet = Copoet()
ai_poet.create_input('I believe')
ai_poet.introduce_rule(('num_verses', 2))
#ai_poet.introduce_rule(('verse_size', 8))
ai_poet.introduce_rule(('num_syl', 18))
#ai_poet.introduce_rule(('cos_sim', 'book', 10, 8))
ai_poet.generate_text()