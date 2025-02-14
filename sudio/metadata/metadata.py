

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



class AudioMetadata:
    """
    Flexible metadata container for audio records.

    Allows dynamic attribute creation and provides dictionary-like 
    access to metadata with additional utility methods.
    """
    def __init__(self, name, **kwargs):
        """
        Initialize an audio metadata object.

        Args
        ----

            - name (str): Primary identifier for the audio record.
            - **kwargs: Additional metadata attributes to set.
        """

        self.name = name
        self.__dict__.update(kwargs)

    def __getitem__(self, key):
        """
        Allow dictionary-style attribute access.

        Args:
        -----

            key (str): Attribute name to retrieve.

        Returns:
        --------

            Value of the specified attribute.
        """
        return getattr(self, key)

    def __setitem__(self, key, value):
        """
        Allow dictionary-style attribute setting.

        Args:
        -----

            - key (str): Attribute name to set.
            - value: Value to assign to the attribute.
        """
        setattr(self, key, value)

    def keys(self):
        """
        Get a list of all non-private attribute names.

        Returns:
        --------

            public attributes in the metadata.
        """
        return [attr for attr in self.__dict__ if not attr.startswith('_')]

    def copy(self):
        """
        Create a deep copy of the metadata object.

        Returns:
        --------

           new instance with the same attributes.
        """
        return AudioMetadata(**{k: getattr(self, k) for k in self.keys()})

    def get_data(self):
        """
        Return the metadata object itself.

        Useful for maintaining a consistent interface with other data retrieval methods.

        Returns:
        --------

            current metadata instance.
        """
        return self