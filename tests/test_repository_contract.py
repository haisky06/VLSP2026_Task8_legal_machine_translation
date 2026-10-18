import ast, re, unittest
from pathlib import Path
from restore_from_qwen_layout import transfer
from translation_safety import split_source, SAFETY_VERSION

ROOT=Path(__file__).resolve().parents[1]
class ContractTests(unittest.TestCase):
    def test_sources_parse_and_have_no_modal_sdk_import(self):
        for path in (ROOT/'vinai_modal').glob('*.py'):
            tree=ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):self.assertFalse(any(n.name=='modal' or n.name.startswith('modal.') for n in node.names))
                if isinstance(node,ast.ImportFrom):self.assertFalse((node.module or '').startswith('modal'))
    def test_source_partition(self):
        text='Article 1. Scope\n\n  1. The parties shall comply.\n'*15
        parts=split_source(text,lambda s:len(s.encode())+2,128)
        self.assertEqual(''.join(p['raw'] for p in parts),text)
        self.assertEqual(SAFETY_VERSION,'exact_source_retry_v2')
    def test_formatter_preserves_generated_content(self):
        original='Article 1. Scope. The parties shall comply.'
        reference='  Article 1. Scope.\n\n  The parties shall comply.'
        formatted,audit=transfer(reference,original)
        self.assertEqual(re.findall(r'\S+',original),re.findall(r'\S+',formatted))
        self.assertTrue(audit['word_tokens_identical'])
        self.assertFalse(audit['exact_source_layout_certified'])

if __name__=='__main__':unittest.main()
