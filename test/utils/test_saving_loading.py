import unittest
from unittest import mock

from academia.utils import SavableLoadable


@mock.patch.multiple(SavableLoadable, __abstractmethods__=frozenset())
class TestSavableLoadable(unittest.TestCase):

    def test_get_type_name_from_class(self):
        # arrange
        sut = SavableLoadable()
        # act
        returned_type_name = sut.get_type_name_full(SavableLoadable)
        # assert
        expected_type_name = 'academia.utils.saving_loading.SavableLoadable'
        self.assertEqual(expected_type_name, returned_type_name)

    def test_get_type_from_name(self):
        # arrange
        sut = SavableLoadable()
        # act
        returned_type = sut.get_type('academia.utils.SavableLoadable')
        # assert
        self.assertEqual(SavableLoadable, returned_type)


if __name__ == '__main__':
    unittest.main()
