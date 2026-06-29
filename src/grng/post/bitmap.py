import numpy as np
from PIL import Image

def bitmap256(data: np.ndarray, output_file: str):

    if data.size != 32: 
        raise ValueError(
            f"Expected exactly 32 bytes, got {len(data)} bytes."
        )

    # Convert bytes -> 256 bits
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))

    # 16x16 image
    img = (1 - bits.reshape((16, 16))).astype(np.uint8) * 255 
    img = Image.fromarray(img, mode="L")
    img = img.resize((256, 256), Image.Resampling.NEAREST)
    img.save(output_file)