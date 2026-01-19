import hashlib
import json
import time


class Block:
    def __init__(self, index, previous_hash, timestamp, data, hash):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.hash = hash


class Blockchain:
    def __init__(self):
        self.chain = []
        self.create_genesis_block()

    def create_genesis_block(self):
        """Create the first block in the blockchain."""
        genesis_block = Block(0, "0", time.time(), "Genesis Block",
                              self.calculate_hash(0, "0", time.time(), "Genesis Block"))
        self.chain.append(genesis_block)

    def add_block(self, data):
        """Add a block to the blockchain."""
        previous_block = self.chain[-1]
        new_index = previous_block.index + 1
        new_timestamp = time.time()
        new_hash = self.calculate_hash(new_index, previous_block.hash, new_timestamp, data)
        new_block = Block(new_index, previous_block.hash, new_timestamp, data, new_hash)
        self.chain.append(new_block)

    def calculate_hash(self, index, previous_hash, timestamp, data):
        """Calculate the hash of the block using SHA-256."""
        block_string = f"{index}{previous_hash}{timestamp}{data}"
        return hashlib.sha256(block_string.encode('utf-8')).hexdigest()

    def print_blockchain(self):
        """Print the blockchain."""
        for block in self.chain:
            print(f"Block #{block.index} [Hash: {block.hash}]")
            print(f"  Previous Hash: {block.previous_hash}")
            print(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(block.timestamp))}")
            print(f"  Data: {json.dumps(block.data)}")
            print()


class IdentityVerificationSystem:
    def __init__(self):
        self.blockchain = Blockchain()

    def verify_identity(self, name, id_number, additional_info):
        """Simulate identity verification and add it as a block to the blockchain."""
        identity_data = {
            "name": name,
            "id_number": id_number,
            "additional_info": additional_info
        }
        self.blockchain.add_block(identity_data)
        print(f"Identity verification for {name} ({id_number}) has been recorded on the blockchain.")

    def show_identity_verifications(self):
        """Show all identity verifications stored on the blockchain."""
        self.blockchain.print_blockchain()


# Example usage
if __name__ == "__main__":
    identity_system = IdentityVerificationSystem()

    # Verify some identities
    identity_system.verify_identity("Alice", "ID12345", {"address": "123 Main St", "dob": "1990-01-01"})
    identity_system.verify_identity("Bob", "ID67890", {"address": "456 Elm St", "dob": "1985-02-28"})

    # Show the blockchain with identity verifications
    identity_system.show_identity_verifications()
