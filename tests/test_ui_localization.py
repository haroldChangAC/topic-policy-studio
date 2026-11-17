import unittest
from html import unescape

from streamlit.testing.v1 import AppTest


class LocalizationTests(unittest.TestCase):
    def test_source_view_preserves_boundary_whitespace_and_escapes_markup(self):
        source = '\n  <source> & “引文”\r\n'
        app = AppTest.from_string(
            'from studio.ui_layout import source_text\n' + f'source_text({source!r})\n'
        ).run()
        self.assertFalse(app.exception)
        rendered = app.html[0].value
        self.assertNotIn('<source>', rendered)
        self.assertEqual(unescape(rendered.split('<span>', 1)[1].split('</span>', 1)[0]), source)
