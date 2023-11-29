import unittest

from academia.environments import MsPacman


class TestGenericAtariWrapper(unittest.TestCase):

    def setUp(self):
        self.sut_stacked3_step = MsPacman(
            difficulty=2,
            n_frames_stacked=3,
            append_step_count=True,
            flatten_state=True,
            random_state=42
        )
        self.sut_step = MsPacman(
            difficulty=2,
            append_step_count=True,
            flatten_state=True,
            random_state=42
        )
        self.sut_basic_flatten = MsPacman(
            difficulty=2,
            flatten_state=True,
            random_state=42
        )
        self.sut_basic = MsPacman(
            difficulty=2,
            random_state=42
        )

    def test_observe(self):
        state_stacked3_step = self.sut_stacked3_step.reset()
        self.assertEqual((302401,), state_stacked3_step.shape)

        state_step = self.sut_step.reset()
        self.assertEqual((100801,), state_step.shape)

        state_basic_flatten = self.sut_basic_flatten.reset()
        self.assertEqual((100800,), state_basic_flatten.shape)

        state_basic = self.sut_basic.reset()
        self.assertEqual((3, 210, 160), state_basic.shape)

    def test_observe_state_obs_type(self):
        sut_graycale_basic = MsPacman(difficulty=2,
                                  random_state=42,
                                  obs_type='grayscale')

        sut_rgb_basic = MsPacman(difficulty=2,
                             random_state=42,
                             obs_type='rgb')

        sut_ram_basic = MsPacman(difficulty=2,
                             random_state=42,
                             obs_type='ram')

        state_grayscale = sut_graycale_basic.reset()
        self.assertEqual((1, 210, 160), state_grayscale.shape)

        state_rgb = sut_rgb_basic.reset()
        self.assertEqual((3, 210, 160), state_rgb.shape)

        state_ram = sut_ram_basic.reset()
        self.assertEqual((128,), state_ram.shape)


if __name__ == '__main__':
    unittest.main()
