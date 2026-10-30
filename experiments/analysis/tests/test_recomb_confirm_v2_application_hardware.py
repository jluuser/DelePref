import unittest
import shutil
import tempfile
from pathlib import Path
from recomb_confirm_v2_application_hardware import assign_hardware,remap_catalog,digest,generate_sources

class HardwareRoutingTests(unittest.TestCase):
    def test_whole_WT_is_routed_by_largest_input_not_method_outcome(self):
        rows=[dict(protein_id='small',physical_id='a'),dict(protein_id='large',physical_id='b'),dict(protein_id='large',physical_id='c')]
        route,size=assign_hardware(rows,dict(a=400,b=700,c=1800))
        self.assertEqual(route,dict(small='A40',large='A100'))
        self.assertEqual(size['large'],1800)

    def test_shared_inputs_close_the_WT_hardware_group(self):
        rows=[dict(protein_id='a',physical_id='shared'),dict(protein_id='b',physical_id='shared'),dict(protein_id='b',physical_id='long')]
        route,_=assign_hardware(rows,dict(shared=400,long=2000))
        self.assertEqual(route,dict(a='A100',b='A100'))

    def test_threshold_is_inclusive_and_input_only(self):
        rows=[dict(protein_id='a',physical_id='x'),dict(protein_id='b',physical_id='y')]
        route,_=assign_hardware(rows,dict(x=1024,y=1025))
        self.assertEqual(route,dict(a='A40',b='A100'))

    def test_hardware_rekeys_identity_without_changing_model_input_or_seed(self):
        profile=dict(device_class='A100_80GB',recycling_steps=3,sampling_steps=200,MSA_sha='real')
        physical=[dict(task_id='old',profile_name='p',profile_sha256=digest(profile),sequence_sha256='seq',weights_sha256='weights',seed=17,sample=0,model_input_sha256='actual',chains=[dict(sequence='AAA')])]
        logical=[dict(request_id='logical',physical_id='old',protein_id='a',method='rag_unlimited',budget=.2,evaluation_role='primary20pct',seed=17)]
        request,rows,profiles,mapping=remap_catalog(physical,logical,dict(p=profile),dict(a='A40'))
        self.assertNotEqual(request[0]['profile_sha256'],physical[0]['profile_sha256'])
        self.assertEqual(request[0]['model_input_sha256'],'actual')
        self.assertEqual(request[0]['chains'],physical[0]['chains'])
        self.assertEqual((request[0]['seed'],request[0]['sample']),(17,0))
        self.assertEqual(rows[0]['physical_id'],mapping['old'])
        self.assertEqual(profiles[request[0]['profile_name']]['sampling_steps'],200)

    def test_mixed_assignment_for_same_shared_prediction_is_rejected(self):
        physical=[dict(task_id='x')]
        rows=[dict(protein_id='a',physical_id='x'),dict(protein_id='b',physical_id='x')]
        with self.assertRaises(ValueError):remap_catalog(physical,rows,{},dict(a='A40',b='A100'))

    def test_real_parent_sources_generate_without_changing_metric_functions(self):
        original=Path(__file__).resolve().parents[2]
        names=['recomb_confirm_v2_application_predict.py','recomb_confirm_v2_application_evaluate.py',
               'recomb_confirm_v2_application_repeat_report.py']
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'analysis').mkdir()
            for name in names:shutil.copyfile(original/'analysis'/name,root/'analysis'/name)
            paths=generate_sources(root)
            self.assertEqual(len(paths),3)
            worker=(root/paths[0]).read_text()
            self.assertIn('a.device_class not in torch.cuda.get_device_name()',worker)
            self.assertIn('torch.cuda.is_bf16_supported()',worker)
            self.assertIn("'boltz_'+a.phase+'_'+a.device_class+'_shard_'",worker)
            for name in names:
                self.assertEqual((root/'analysis'/name).read_bytes(),(original/'analysis'/name).read_bytes())

if __name__=='__main__':unittest.main()

