/* eslint-disable no-console */
import { Injectable } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { lastValueFrom } from 'rxjs';
import { CreateNewSongDto } from './dto/create-new-song.dto';
import { ConfigService } from '@nestjs/config';
import { YtService } from './ytService';
import { DownloadService } from './downloadService';
import { GenreService } from './genreService';
import { SongStorageService } from './new-songs.service';
import { Cron } from '@nestjs/schedule';

@Injectable()
export class SpotifyService {
  private readonly spotifyApiUrl =
    this.configService.get<string>('SPOTIFY_API_URL');

  constructor(
    private readonly httpService: HttpService,
    private readonly configService: ConfigService,
    private readonly ytService: YtService,
    private readonly downloadService: DownloadService,
    private readonly genreService: GenreService,
    private readonly songService: SongStorageService,
  ) { }

  async getAccessToken(): Promise<string> {
    const clientId = this.configService.get<string>('SPOTIFY_CLIENT_ID');
    const clientSecret = this.configService.get<string>(
      'SPOTIFY_CLIENT_SECRET',
    );
    const { data } = await lastValueFrom(
      this.httpService.post('https://accounts.spotify.com/api/token', null, {
        params: {
          grant_type: 'client_credentials',
        },
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
          Authorization: `Basic ${Buffer.from(`${clientId}:${clientSecret}`).toString('base64')}`,
        },
      }),
    );
    return data.access_token;
  }

  @Cron('0 0 * * 5')
  async handleCron() {
    const newSongsDto = await this.getNewReleasedSongs();
    console.log(newSongsDto);
  }

  async getNewReleasedSongs(): Promise<CreateNewSongDto> {
    const playlistId = this.configService.get<string>('PLAYLIST_ID');
    const limit = 15; // Desired number of songs
    const accessToken = await this.getAccessToken();

    const { data } = await lastValueFrom(
      this.httpService.get(
        `${this.spotifyApiUrl}/playlists/${playlistId}/tracks`,
        {
          params: {
            limit,
            fields:
              'items(track(name,album(artists.name,images),external_urls.spotify))',
          },
          headers: {
            Authorization: `Bearer ${accessToken}`,
          },
        },
      ),
    );

    const songs = await Promise.all(
      data.items.map(async (item) => {
        const name = item.track.name;
        const artistName = item.track.album.artists[0].name;
        const coverImage = item.track.album.images[0].url;
        const spotifyLink = item.track.external_urls.spotify;
        const m4aLink = await this.ytService.getM4aLink(name);
        const wavFilePath = await this.downloadService.downloadAndConvertToWav(
          name,
          m4aLink,
        );
        return {
          name,
          ytLink: m4aLink,
          wavFilePath,
          artistName,
          coverImage,
          spotifyLink,
        };
      }),
    );

    const newSongDto: CreateNewSongDto = {
      songs,
      date: new Date(),
    };

    // Get the predicted genres for the songs
    const songsWithGenre =
      await this.genreService.getGenresForSongs(newSongDto);
    return await this.songService.saveSongs(songsWithGenre);
  }
}
