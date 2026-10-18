"""拒绝数字设备绑定和错误设备。"""

from types import SimpleNamespace
import unittest

from rsi.devices import require_uuid_binding, verify_device
from rsi.validation import ValidationError


UUID = 'GPU-bebc6ef5-f32c-961a-ca6d-62b5da7af1c6'


class DeviceTests(unittest.TestCase):
    def test_numeric_or_multiple_device_bindings_are_rejected(self) -> None:
        for value in ('0','5',UUID + ',' + UUID,None):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                require_uuid_binding('cuda:0',value)

    def test_single_uuid_uses_logical_zero(self) -> None:
        self.assertEqual(require_uuid_binding('cuda:0',UUID),UUID)
        with self.assertRaises(ValidationError):
            require_uuid_binding('cuda:5',UUID)

    def test_wrong_hardware_name_or_uuid_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            verify_device(SimpleNamespace(name='NVIDIA GeForce RTX 3090',uuid=UUID,total_memory=24),
                          'NVIDIA RTX 5880 Ada Generation',UUID)
        with self.assertRaises(ValidationError):
            verify_device(SimpleNamespace(name='NVIDIA RTX 5880 Ada Generation',uuid='other',total_memory=48),
                          'NVIDIA RTX 5880 Ada Generation',UUID)

    def test_verified_hardware_identity_is_recorded(self) -> None:
        result = verify_device(SimpleNamespace(name='5880',uuid=UUID[4:],total_memory=48),'5880',UUID)
        self.assertEqual(result['uuid'],UUID)
