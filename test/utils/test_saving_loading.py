import os
import tempfile

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

    def test_prep_save_file_dirs_created(self):
        # arrange
        with tempfile.TemporaryDirectory() as temp_dir:
            dirname = os.path.join(temp_dir, 'dir1')
            specified_save_path = os.path.join(dirname, 'file.txt')
            # act
            SavableLoadable.prep_save_file(specified_save_path, interrupted=False)
            # assert
            self.assertTrue(os.path.isdir(dirname), msg='Intermediate directories should get created')

    def test_prep_save_file_name_normal(self):
        # arrange
        with tempfile.TemporaryDirectory() as temp_dir:
            dirname = os.path.join(temp_dir, 'dir1')
            specified_save_path = os.path.join(dirname, 'file.txt')
            # act
            actual_save_path = SavableLoadable.prep_save_file(specified_save_path, interrupted=False)
            # assert
            actual_save_filename = os.path.split(actual_save_path)[1]
            self.assertEqual('file.txt', actual_save_filename,
                             msg='Filename should be exactly the same as specified')

    def test_prep_save_file_name_interrupted(self):
        # arrange
        with tempfile.TemporaryDirectory() as temp_dir:
            dirname = os.path.join(temp_dir, 'dir1')
            specified_save_path = os.path.join(dirname, 'file.txt')
            # act
            actual_save_path = SavableLoadable.prep_save_file(specified_save_path, interrupted=True)
            # assert
            actual_save_filename = os.path.split(actual_save_path)[1]
            self.assertEqual('backup_file.txt', actual_save_filename,
                             msg='Filename should have a "backup_" prefix prepended')


if __name__ == '__main__':
    unittest.main()
