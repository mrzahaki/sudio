

# SUDIO - Audio Processing Platform
# Copyright (C) 2024 Hossein Zahaki

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
#  any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# - GitHub: https://github.com/MrZahaki/sudio



class AudioRecordDatabase:
    """
    lightweight in-memory database for managing audio records. provides dictionary-like access and manipulation of audio records
    with simple key (name) based storage and retrieval.
    """

    def __init__(self):
        """
        Initialize an empty audio record database.
        """
        self.records = {}

    def add_record(self, record):
        """
        Add an audio record to the database.

        Args:
        -----

            record: An audio record with a 'name' attribute to be used as the key.
        """
        self.records[record.name] = record

    def get_record(self, name):
        """
        Retrieve a specific audio record by its name.

        Args:
        -----
            
            name (str): Name of the audio record to retrieve.
        """

        return self.records.get(name)

    def remove_record(self, name):
        """
        Remove a record from the database by its name.

        Args:
        -----

            name (str): Name of the audio record to remove.
        """
        if name in self.records:
            del self.records[name]

    def index(self):
        """
        Get a list of all record names in the database.

        Returns:
        --------

            list: Names of all records currently in the database.
        """
        return list(self.records.keys())

    def __getitem__(self, name):
        """
        Allow dictionary-style access to records.

        Args:
        -----

            name (str): Name of the record to retrieve.

        """
        return self.get_record(name)

    def __setitem__(self, name, record):
        """
        Allow dictionary-style record addition.

        Args:
        -----

            - name (str): Name to associate with the record.
            - record: The audio record to store.
        """
        self.add_record(record)