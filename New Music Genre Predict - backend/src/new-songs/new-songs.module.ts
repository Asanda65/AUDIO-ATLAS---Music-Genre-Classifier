import { Module } from '@nestjs/common';
import { SongStorageService } from './new-songs.service';
import { NewSongsController } from './new-songs.controller';
import { SpotifyService } from './SpotifyService';
import { ConfigModule } from '@nestjs/config';
import { HttpModule } from '@nestjs/axios';
import { YtService } from './ytService';
import { DownloadService } from './downloadService';
import { GenreService } from './genreService';
import { MongooseModule } from '@nestjs/mongoose';
import { NewSongSchema } from './entities/new-song.entity';
import { ScheduleModule } from '@nestjs/schedule';

@Module({
  imports: [
    ScheduleModule.forRoot(),
    MongooseModule.forFeature([{ name: 'NewSong', schema: NewSongSchema }]),
    ConfigModule,
    HttpModule,
  ],
  controllers: [NewSongsController],
  providers: [
    SongStorageService,
    SpotifyService,
    YtService,
    DownloadService,
    GenreService,
  ],
})
export class NewSongsModule {}
