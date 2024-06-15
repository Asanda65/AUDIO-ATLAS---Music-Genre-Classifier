import { Controller, Get } from '@nestjs/common';
import { SpotifyService } from './SpotifyService';
import { NewSong } from './entities/new-song.entity';
import { SongStorageService } from './new-songs.service';

@Controller('new-songs')
export class NewSongsController {
  constructor(
    private readonly spotifyService: SpotifyService,
    private readonly newSongsService: SongStorageService,
  ) {}

  @Get('latest')
  async getLatestSongs(): Promise<NewSong> {
    return this.newSongsService.getLatestSongs();
  }

  @Get()
  findAll() {
    return this.spotifyService.getNewReleasedSongs();
  }
}
