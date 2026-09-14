import base64
import os

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


def get_keys_path(role):
    # Get the current directory of the script
    keys_dir = os.path.join(os.getcwd(), f"keys/{role}")

    # Create the keys directory if it doesn't exist
    os.makedirs(keys_dir, exist_ok=True)

    return keys_dir


def generate_rsa_key_pair(role) -> None:
    path = get_keys_path(role)

    if not os.path.exists(os.path.join(path, "public_key.pem")):
        # Save private and public rsa keys to files
        with open(os.path.join(path, "private_key.pem"), "wb") as f:
            key = rsa.generate_private_key(
                public_exponent=65537, key_size=2048, backend=default_backend()
            )

            f.write(
                key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )

        with open(os.path.join(path, "public_key.pem"), "wb") as f:
            f.write(
                key.public_key().public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            )


def load_public_key(role):
    path = get_keys_path(role)
    path = os.path.join(path, "public_key.pem")
    generate_rsa_key_pair(role)

    with open(path, "rb") as f:
        return serialization.load_pem_public_key(f.read(), backend=default_backend())


def load_private_key(role):
    path = get_keys_path(role)
    path = os.path.join(path, "private_key.pem")
    generate_rsa_key_pair(role)

    with open(path, "rb") as f:
        return serialization.load_pem_private_key(
            f.read(), backend=default_backend(), password=None
        )


def authenticate_public_key(public_key) -> bool:
    try:
        public_key = serialization.load_pem_public_key(
            public_key, backend=default_backend()
        )

        if public_key.public_numbers().e != 65537:
            return False

        if not isinstance(public_key, rsa.RSAPublicKey):
            return False

        return True

    except Exception as e:
        return False


def get_public_key_bytes(public_key):
    public_key_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return public_key_bytes


def get_private_key_bytes(private_key):
    private_key_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
    )
    return private_key_bytes


def get_public_key_obj(public_key_bytes):
    return serialization.load_pem_public_key(
        public_key_bytes, backend=default_backend()
    )


def get_private_key_obj(private_key_bytes):
    return serialization.load_pem_private_key(
        private_key_bytes, backend=default_backend(), password=None
    )


def get_rsa_pub_key(role, b=False):
    public_key = load_public_key(role)

    if b is True:
        return get_public_key_bytes(public_key)
    else:
        return public_key


def get_rsa_priv_key(role, b=False):
    private_key = load_private_key(role)

    if b is True:
        return get_private_key_bytes(private_key)
    else:
        return private_key


def encrypt(data, role, pub_key: bytes = None):
    # Encrypt the data using RSA-OAEP
    if pub_key is None:
        pub_key = get_rsa_pub_key(role)
    else:
        pub_key = get_public_key_obj(pub_key)

    encrypted_data = pub_key.encrypt(
        data,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )

    return base64.b64encode(encrypted_data)


def decrypt(data, role):
    private_key = get_rsa_priv_key(role)

    decrypted_data = private_key.decrypt(
        base64.b64decode(data),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )

    return decrypted_data
