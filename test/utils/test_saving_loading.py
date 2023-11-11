import unittest
from unittest import mock
from academia.utils import SavableLoadable


@mock.patch.multiple(SavableLoadable, __abstractmethods__=frozenset())
class MyTestCase(unittest.TestCase):

    def test_get_type_name_from_class(self):
        # arrange
        spt = SavableLoadable()
        # act
        returned_type_name = spt.get_type_name_full(SavableLoadable)
        # assert
        expected_type_name = 'academia.utils.saving_loading.SavableLoadable'
        self.assertEqual(returned_type_name, expected_type_name)

    def test_get_type_from_name(self):
        # arrange
        spt = SavableLoadable()
        # act
        returned_type = spt.get_type('academia.utils.SavableLoadable')
        # assert
        self.assertEqual(returned_type, SavableLoadable)


if __name__ == '__main__':
    unittest.main()
