interface Song {
  name: string;
  ytLink?: string;
  wavFilePath?: string;
  genre?: string;
  artistName?: string;
  coverImage?: string;
  spotifyLink?: string;
}

export class CreateNewSongDto {
  songs: Song[];
  date?: Date;
}
