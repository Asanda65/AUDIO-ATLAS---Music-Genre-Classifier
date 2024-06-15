import { PartialType } from '@nestjs/swagger';
import { CreateNewSongDto } from './create-new-song.dto';

export class UpdateNewSongDto extends PartialType(CreateNewSongDto) {}
