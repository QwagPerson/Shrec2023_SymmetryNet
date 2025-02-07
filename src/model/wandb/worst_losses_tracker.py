class WorstLossesTracker:
	def __init__(self, max_entries=100):
		self.max_entries = max_entries
		self.entries = []
		self.required_keys = {
			'fn', 'loss', 'batch_idx', 'batch', 
			'train_val_test_tag', 'class_id', 'plane_predictions'
		}

	def add(self, entry):
		"""Add a new entry to the tracker if it qualifies as a worst loss."""
		# Validate required keys
		if not self._validate_entry(entry):
			missing = [k for k in self.required_keys if k not in entry]
			raise ValueError(f"Entry missing required keys: {missing}")
		
		# Short-circuit if list is full and new loss isn't large enough
		if self.entries and len(self.entries) >= self.max_entries:
			min_loss = self.entries[-1]['loss']
			if entry['loss'] <= min_loss:
				return

		# Find insertion point
		insert_idx = 0
		while insert_idx < len(self.entries) and entry['loss'] <= self.entries[insert_idx]['loss']:
			insert_idx += 1

		# Insert and maintain size
		self.entries.insert(insert_idx, entry)
		if len(self.entries) > self.max_entries:
			self.entries.pop()

	def _validate_entry(self, entry):
		"""Ensure the entry contains all required keys."""
		return all(key in entry for key in self.required_keys)

	def empty(self):
		"""Clear all entries from the tracker."""
		self.entries.clear()

	def remove(self, index):
		"""Remove an entry by its index."""
		if 0 <= index < len(self.entries):
			del self.entries[index]

	def get_entries(self):
		"""Get a copy of the current list of worst losses."""
		return self.entries.copy()

	def __len__(self):
		"""Current number of tracked entries."""
		return len(self.entries)

	def set_max_entries(self, max_entries):
		"""Update the maximum number of entries (truncates if needed)."""
		self.max_entries = max_entries
		while len(self.entries) > self.max_entries:
			self.entries.pop()
