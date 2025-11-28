import sys
sys.modules['annoy'] = __import__('fake_annoy')

from fashion_clip.fashion_clip import FashionCLIP
model = FashionCLIP('fashion-clip')

print("✓ FashionCLIP is working")
