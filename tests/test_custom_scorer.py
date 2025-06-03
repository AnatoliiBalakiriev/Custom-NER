import ast
import types


def load_custom_scorer_class():
    with open('custom_scorer_2_copy.py', 'r', encoding='utf-8') as f:
        source = f.read()
    module_ast = ast.parse(source)
    class_def = next(
        node for node in module_ast.body if isinstance(node, ast.ClassDef) and node.name == 'Custom_Scorer'
    )
    module = types.ModuleType('temp_custom_scorer')
    exec(compile(ast.Module([class_def], []), 'custom_scorer_2_copy.py', 'exec'), module.__dict__)
    return module.Custom_Scorer


Custom_Scorer = load_custom_scorer_class()


def test_convert_to_spacy_format_simple():
    sample_json = [
        {
            "text": "John loves Mary",
            "label": [
                {"start": 0, "end": 4, "labels": ["PERSON"]},
                {"start": 11, "end": 15, "labels": ["PERSON"]},
            ],
        }
    ]

    scorer = Custom_Scorer.__new__(Custom_Scorer)
    result = Custom_Scorer.convert_to_spacy_format(scorer, sample_json)
    expected = [(
        "John loves Mary",
        {"entities": [(0, 4, "PERSON"), (11, 15, "PERSON")]},
    )]

    assert result == expected
