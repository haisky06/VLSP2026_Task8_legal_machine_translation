import tempfile,unittest,csv,io,zipfile
from pathlib import Path
from submission_export import export_submission

class ExportTests(unittest.TestCase):
    def test_roundtrip(self):
        expected=[dict(sample_id=str(i),direction='en-vi' if i<50 else 'vi-en') for i in range(100)]
        translations={d:{} for d in ('en-vi','vi-en')}
        text='Điều 1, "Nghĩa vụ"\n1. Parties shall comply.\n'
        for row in expected:translations[row['direction']][row['sample_id']]=text
        translations['en-vi']['0']=text+'A'*200000
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            result=export_submission(folder,translations,expected)
            with zipfile.ZipFile(result['zip']) as z:
                self.assertEqual(z.namelist(),['results.csv'])
                rows=list(csv.DictReader(io.StringIO(z.read('results.csv').decode('utf8'),newline='')))
                self.assertEqual(len(rows),100)
                self.assertEqual(rows[0]['translation'],text+'A'*200000)
                self.assertTrue(all(r['translation']==text for r in rows[1:]))
            translations['en-vi']['0']=''
            with self.assertRaises(RuntimeError):export_submission(folder,translations,expected)
    def test_duplicate_expected(self):
        with self.assertRaises(RuntimeError):export_submission(Path(__file__).parent,{},[dict(sample_id='a',direction='vi-en')]*100)

if __name__=='__main__':unittest.main()
