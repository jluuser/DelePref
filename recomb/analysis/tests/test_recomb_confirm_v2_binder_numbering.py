import unittest
from recomb_confirm_v2_binder_numbering import domain_summary


class NumberingMappingTests(unittest.TestCase):
    def test_CDR3_insertions_keep_sequence_order_and_exclude_anchors(self):
        numbered=[((104,' '),'C'),((105,' '),'A'),((111,'A'),'D'),((112,'B'),'E'),((117,' '),'F'),((118,' '),'W')]
        d=domain_summary('QCADEFWQ',(numbered,1,6),dict(chain_type='H'))
        self.assertEqual(d['CDR3'],'ADEF')
        self.assertEqual([p['WT_index'] for p in d['CDR3_WT_positions']],[2,3,4,5])

    def test_wrong_domain_offset_does_not_silently_map(self):
        with self.assertRaises(ValueError):domain_summary('ACDE',([((105,' '),'C')],0,0),dict(chain_type='H'))

    def test_alignment_gaps_do_not_consume_input_positions(self):
        d=domain_summary('QACWQ',([((104,' '),'A'),((105,' '),'C'),((106,' '),'-'),((118,' '),'W')],1,3),dict(chain_type='H'))
        self.assertEqual(d['CDR3_WT_positions'][0]['WT_index'],2)


if __name__=='__main__':unittest.main()
